import os, sys
import yaml
import multiprocessing
from functools import partial

from collections import defaultdict

import torch
from torch.utils.data import ConcatDataset
from torchvision import transforms

import cv2
import numpy as np

from nmbg.datasets.common import ToTensor, load_image, get_dataset_config, split_lists
from nmbg.gl.utils import get_proj_matrix, load_scene_data, FastRand
from nmbg.utils.perform import TicToc, AccumDict
from pathlib import Path
import itertools
import random


def rescale_K(K_, sx, sy, keep_fov=True):
    K = K_.copy()
    K[0, 2] = sx * K[0, 2]
    K[1, 2] = sy * K[1, 2]
    if keep_fov:
        K[0, 0] = sx * K[0, 0]
        K[1, 1] = sy * K[1, 1]
    return K


def rand_(min_, max_, *args):
    return min_ + (max_ - min_) * np.random.rand(*args)


default_input_transform = transforms.Compose([
        ToTensor(),
])

default_target_transform = transforms.Compose([
        ToTensor(),
])

def get_rnd_crop_center_v1(mask, factor=8):
    mask_down = mask[::factor, ::factor]
    foregr_i, foregr_j = np.nonzero(mask_down)
    pnt_idx = np.random.choice(len(foregr_i))
    pnt = (foregr_i[pnt_idx] * factor, foregr_j[pnt_idx] * factor)
    return pnt


class DynamicDataset:
    znear = 0.1
    zfar = 1000
    
    def __init__(self, scene_data, image_size,
                 view_list, target_list, mask_list, label_list,
                 rotation_matrix_ext_list, rotation_matrix_cam_list,
                 translation_vec_ext_list,translation_vec_cam_list, 
                 keep_fov=False,
                 input_transform=None, target_transform=None,
                 num_samples=None,
                 random_zoom=None, random_shift=None,
                 drop_points=0., perturb_points=0.,
                 label_in_input=False,
                 crop_by_mask=False,
                 theta=None, phi=None, 
                 center = 0,
                 radius = 0,
                 supersampling=1):
        if isinstance(image_size, (int, float)):
            image_size = image_size, image_size
        
        # if render image size is different from camera image size, then shift principal point
        K_src = scene_data['intrinsic_matrix']
        old_size = scene_data['config'] 
        sx = image_size[0] / old_size[0]
        sy = image_size[1] / old_size[1]
        K = rescale_K(K_src, sx, sy, keep_fov)
        
        assert len(view_list) == len(target_list)
        
        print('image_size', image_size)
        self.view_list = view_list
        self.target_list = target_list
        self.mask_list = mask_list
        self.label_list = label_list
        self.rotation_matrix_ext_list = rotation_matrix_ext_list
        self.rotation_matrix_cam_list = rotation_matrix_cam_list
        self.translation_vec_ext_list = translation_vec_ext_list
        self.translation_vec_cam_list = translation_vec_cam_list
        self.scene_data = scene_data
        self.image_size = image_size
        self.old_image_size = old_size
        self.renderer = None
        self.scene = None
        self.K = K
        self.K_src = K_src
        self.random_zoom = random_zoom
        self.random_shift = random_shift
        self.sx = sx
        self.sy = sy
        self.keep_fov = keep_fov
        self.target_list = target_list
        self.input_transform = default_input_transform if input_transform is None else input_transform
        self.target_transform = default_target_transform if target_transform is None else target_transform
        self.num_samples = num_samples if num_samples else len(view_list)
        self.id = None
        self.name = None
        self.drop_points = drop_points
        self.perturb_points = perturb_points
        self.label_in_input = label_in_input
        self.crop_by_mask = crop_by_mask
        self.theta = theta
        self.phi = phi
        self.center = center
        self.radius = radius
        self.ss = supersampling

        self.fastrand = None
        self.timing = None
        self.count = 0
        if isinstance(self.theta, np.ndarray):
            self.sampling_rate = len(view_list)//round(0.02*len(view_list))

    def normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def load(self):
        if self.perturb_points and self.fastrand is None:
            print(f'SETTING PERTURB POINTS: {self.perturb_points}')
            tform = lambda p: self.perturb_points * (p - 0.5)
            self.fastrand = FastRand((self.scene_data['pointcloud']['xyz'].shape[0], 2), tform, 10) 

    def viewmatrix(self, lookdir, up, position, subtract_position=False):
        """Construct lookat view matrix."""
        vec2 = self.normalize((lookdir - position) if subtract_position else lookdir)
        vec0 = self.normalize(np.cross(up, vec2))
        vec1 = self.normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def unload(self):
        print("debug")
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        idx = idx % len(self.view_list)

        if self.timing is None:
            self.timing = AccumDict()

        if idx % self.sampling_rate == 0 and isinstance(self.theta, np.ndarray):
            radius_rand = np.random.uniform(low=0.6*self.radius, high=self.radius, size=(1,))
            look_at_pos = self.center
            if self.theta[idx] > 0: 
                self.theta[idx] = 90 - np.abs(self.theta[idx])
                self.theta[idx] = np.radians(self.theta[idx])
            else: 
                self.theta[idx] = -(90 - np.abs(self.theta[idx]))
                self.theta[idx] = np.radians(self.theta[idx])

            camera_pos = self.center + radius_rand * np.array([np.sin(self.theta[idx]) * np.sin(self.phi[idx]), np.cos(self.theta[idx]) , np.sin(self.theta[idx]) * np.cos(self.phi[idx])])
            scene_id = {}
            scene_id['id'] = self.id
            view_matrix = torch.eye(4)
            cam_trans_vec = np.transpose(camera_pos)
            up = np.array([0., 1., 0.])
            cam_rot_mat = np.transpose(self.viewmatrix(look_at_pos, up, camera_pos, subtract_position=True)[0])[:3,:3]
            ext_rot_mat = np.transpose(cam_rot_mat)
            ext_trans_vec = np.matmul(-ext_rot_mat, cam_trans_vec)
            cam_trans_vec = cam_trans_vec.squeeze(-1)
            ext_trans_vec = ext_trans_vec.squeeze(-1)

            mask = None
            mask_crop = None
            if self.mask_list[idx]:
                mask = load_image(self.mask_list[idx])

                if self.crop_by_mask:
                    cnt = get_rnd_crop_center_v1(mask[..., 0])
                    mask_crop = -1 + 2 * np.array(cnt) / mask.shape[:2]

            K, _, shift_x, shift_y, shift_z = self._get_intrinsics(shift=mask_crop)
            K_pytorch3d, _ = self._get_intrinsics_pytorch3d(shift_x, shift_y, shift_z, shift=mask_crop)

            target = load_image(self.target_list[idx])
            target = self._warp(target, K)
            width = self.image_size[0]
            height = self.image_size[1]

            if mask is None:
                mask = np.ones((target.shape[0], target.shape[1], 1), dtype=np.float32)
            else:
                mask = self._warp(mask, K)

            if self.label_list[idx]:
                label = load_image(self.label_list[idx])
                label = self._warp(label, K)
                label = label[..., :1]
            else:
                label = np.zeros((target.shape[0], target.shape[1], 1), dtype=np.uint8)

            target = self.target_transform(target)
            mask = ToTensor()(mask)
            label = ToTensor()(label)

            ray_directions_img_list = []
            for res in range(5):
                ray_directions_img = self._ray_directions(K, ext_rot_mat, cam_rot_mat, ext_trans_vec, cam_trans_vec, height, width, res)
                ray_directions_img = self.target_transform(ray_directions_img)
                ray_directions_img_list.append(ray_directions_img)

            aug_flag = 1
        
        else:
            tt = TicToc()
            tt.tic()

            mask = None
            mask_crop = None
            if self.mask_list[idx]:
                mask = load_image(self.mask_list[idx])

                if self.crop_by_mask:
                    cnt = get_rnd_crop_center_v1(mask[..., 0])
                    mask_crop = -1 + 2 * np.array(cnt) / mask.shape[:2]

            view_matrix = self.view_list[idx]
            ext_rot_mat = self.rotation_matrix_ext_list[idx]
            cam_rot_mat = self.rotation_matrix_cam_list[idx]
            ext_trans_vec = self.translation_vec_ext_list[idx]
            cam_trans_vec = self.translation_vec_cam_list[idx]
            K, _, shift_x, shift_y, shift_z = self._get_intrinsics(shift=mask_crop)
            K_pytorch3d, _ = self._get_intrinsics_pytorch3d(shift_x, shift_y, shift_z, shift=mask_crop)

            target = load_image(self.target_list[idx])
            target = self._warp(target, K)
            width = self.image_size[0]
            height = self.image_size[1]

            if mask is None:
                mask = np.ones((target.shape[0], target.shape[1], 1), dtype=np.float32)
            else:
                mask = self._warp(mask, K)

            if self.label_list[idx]:
                label = load_image(self.label_list[idx])
                label = self._warp(label, K)
                label = label[..., :1]
            else:
                label = np.zeros((target.shape[0], target.shape[1], 1), dtype=np.uint8)
            
            self.timing.add('get_target', tt.toc())
            tt.tic()

            if self.drop_points:
                self.scene.set_point_discard(np.random.rand(self.scene_data['pointcloud']['xyz'].shape[0]) < self.drop_points)

            if self.perturb_points:
                self.scene.set_point_perturb(self.fastrand.toss())
            
            if self.label_in_input:
                for k in input_:
                    if 'labels' in k:
                        m = input_[k].sum(2) > 1e-9
                        label_sz = cv2.resize(label, (input_[k].shape[1], input_[k].shape[0]), interpolation=cv2.INTER_NEAREST)
                        label_m = label_sz * m
                        input_[k] = label_m[..., None]
            
            self.timing.add('render', tt.toc())
            tt.tic()
            
            target = self.target_transform(target)
            mask = ToTensor()(mask)
            label = ToTensor()(label)

            scene_id = {}
            scene_id['id'] = self.id
            rotation_matrix_ext = np.transpose(self.rotation_matrix_ext_list[idx])
            rotation_matrix_cam = self.rotation_matrix_cam_list[idx]
            trans_vec_ext = self.translation_vec_ext_list[idx]
            trans_vec_cam = self.translation_vec_cam_list[idx]

            ray_directions_img_list = []
            for res in range(5):
                ray_directions_img = self._ray_directions(K, rotation_matrix_ext, rotation_matrix_cam, trans_vec_ext, trans_vec_cam, height, width, res)
                ray_directions_img = self.target_transform(ray_directions_img)
                ray_directions_img_list.append(ray_directions_img)
            
            self.timing.add('transform', tt.toc())
            aug_flag = 0

            self.count += 1
        
        return {'scene_id': scene_id,
                'augment_cams': aug_flag,
                'view_matrix': view_matrix,
                'ext_rot_mat': ext_rot_mat,
                'ext_trans_vec': ext_trans_vec,
                'cam_rot_mat': cam_rot_mat,
                'cam_trans_vec': cam_trans_vec,
                'intrinsic_matrix': K,
                'intrinsic_matrix_pytorch3d': K_pytorch3d,    
                'ray_directions_img': ray_directions_img_list,
                'target': target,
                'target_filename': self.target_list[idx],
                'mask': mask,
                'label': label
               }

    def _get_intrinsics(self, shift=None):
        K = self.K.copy()
        sx = 1. if self.keep_fov else self.sx
        sy = 1. if self.keep_fov else self.sy
        x = 0
        y = 0
        z = 0
        if self.random_zoom:
            z = rand_(*self.random_zoom)
            K[0, 0] *= z
            K[1, 1] *= z
            sx /= z
            sy /= z
        if self.random_shift:
            if shift is None:
                x, y = rand_(*self.random_shift, 2)
            else:
                x, y = shift
            w = self.image_size[0] * (1. - sx) / sx / 2.
            h = self.image_size[1] * (1. - sy) / sy / 2.
            K[0, 2] += x * w
            K[1, 2] += y * h
            
        return K, get_proj_matrix(K, self.image_size, self.znear, self.zfar), x, y, z

    def _get_intrinsics_pytorch3d(self, x, y, z, shift=None):
        K = self.K.copy()
        sx = 1. if self.keep_fov else self.sx
        sy = 1. if self.keep_fov else self.sy
        if self.random_zoom:
            #z = rand_(*self.random_zoom)
            K[0, 0] *= z
            K[1, 1] *= z
            sx /= z
            sy /= z
        if self.random_shift:
            #if shift is None:
                #x, y = rand_(*self.random_shift, 2)
            #else:
                #x, y = shift
            w = self.image_size[0] * (1. - sx) / sx / 2.
            h = self.image_size[1] * (1. - sy) / sy / 2.
            K[0, 2] -= x * w
            K[1, 2] -= y * h
            
        return K, get_proj_matrix(K, self.image_size, self.znear, self.zfar)
    
    def _warp(self, image, K):
        H = K @ np.linalg.inv(self.K_src)
        image = cv2.warpPerspective(image, H, tuple(self.image_size))
        return image

    def _ray_directions(self, intrinsic_matrix_mod, rot_mat_ext, rot_mat_cam, trans_vec_ext, trans_vec_cam, height, width, res):
        res_factor = 2 ** res
        width = width // res_factor
        height = height // res_factor
        u_idxs = (np.array(range(width)) + 1).tolist() 
        v_idxs = (np.array(range(height)) + 1).tolist() 
        uv_values = itertools.product(u_idxs, v_idxs)
        uv_values = np.array(list(uv_values))
        uv_values_original = uv_values.copy() - 1
        uv_values = uv_values.astype(float)
        ones_idx = np.ones_like(uv_values[:,0])

        fx, fy, cx, cy = self.parse_intrinsics(intrinsic_matrix_mod)
        fx = fx / res_factor
        fy = fy / res_factor
        cx = cx / res_factor
        cy = cy / res_factor
        cam_u = ((uv_values[:,0] - cx)/fx)*ones_idx
        cam_v = ((uv_values[:,1] - cy)/fy)*ones_idx
        cam_u_v = np.stack((cam_u, cam_v, ones_idx), axis=-1)
        cam_u_v = np.transpose(cam_u_v)
        world_u_v = np.matmul(rot_mat_cam, cam_u_v)
        ray_directions = world_u_v

        ray_directions_norm = np.expand_dims(np.linalg.norm(ray_directions, axis=0), axis=0)
        ray_directions /= ray_directions_norm
        ray_directions = ray_directions.transpose(1,0)

        ray_directions_img = np.ones((height, width, 3))
        ray_directions_img[uv_values_original[:,1], uv_values_original[:,0], :] = ray_directions 
        return ray_directions_img

    def parse_intrinsics(self, intrinsics):
        fx = intrinsics[..., 0, :1]
        fy = intrinsics[..., 1, 1:2]
        cx = intrinsics[..., 0, 2:3]
        cy = intrinsics[..., 1, 2:3]
        return fx, fy, cx, cy

    def _warp_inv(self, K):
        H = self.K_src @ np.linalg.inv(K)
        return H


def get_datasets(args):
    assert args.paths_file, 'set paths'
    # assert args.dataset_names, 'set dataset_names'

    with open(args.paths_file) as f:
        paths_data = yaml.full_load(f)

    if not args.dataset_names:
        print('Using all datasets')
        args.dataset_names = list(paths_data['datasets'])

    if args.exclude_datasets:
        args.dataset_names = list(set(args.dataset_names) - set(args.exclude_datasets))

    #pool = multiprocessing.Pool(32)
    #map_fn = partial(_load_dataset, paths_data=paths_data, args=args)
    #pool_out = pool.map_async(map_fn, args.dataset_names)
    
    # pool_out = [_load_dataset(tasks[0])]

    ds_train_list, ds_val_list, ds_name = [], [], []

    # for ds_train, ds_val in pool_out.get():
    #     ds_train_list.append(ds_train)
    #     ds_val_list.append(ds_val)

    #     print(f'ds_train: {len(ds_train)}')
    #     print(f'ds_val: {len(ds_val)}')    

    for name in args.dataset_names:
        print(f'creating dataset {name}')
        
        ds_train, ds_val = _get_splits(paths_data, name, args)

        ds_train.name = ds_val.name = name
        ds_train.id = ds_val.id = args.dataset_names.index(name)

        ds_train_list.append(ds_train)
        ds_val_list.append(ds_val)
        ds_name.append(name)

        print(f'ds_train: {len(ds_train)}')
        print(f'ds_val: {len(ds_val)}')

    #pool.close()

    return ds_train_list, ds_val_list, ds_name


def _load_dataset(name, paths_data, args):
    ds_train, ds_val = _get_splits(paths_data, name, args)

    ds_train.name = ds_val.name = name
    ds_train.id = ds_val.id = args.dataset_names.index(name)

    return ds_train, ds_val


def _get_splits(paths_file, ds_name, args):
    config = get_dataset_config(paths_file, ds_name)

    if len(config['val_indices']) > 0:
        val_indices = config['val_indices']

    current_dir = Path().absolute()
    ibr_dir = Path(str(current_dir)  + str(Path(config['target_path'])))
    im_ext = ".jpg"
    im_paths = sorted(ibr_dir.glob(f"im_*{im_ext}"))

    im_paths_new = []
    for idx in range(len(im_paths)):
        im_paths_new.append(str(im_paths[idx]))

    target_list = im_paths_new
    (width, height) = (load_image(target_list[0]).shape[1], load_image(target_list[0]).shape[0])

    scene_data = load_scene_data(ibr_dir)
    view_list = scene_data['view_matrix']

    scene_data['config'] = (width, height)

    rotation_matrix_ext_list = scene_data['rotation_matrix_ext']
    rotation_matrix_cam_list = scene_data['rotation_matrix_cam']
    translation_vec_ext_list = scene_data['translation_vec_ext']
    translation_vec_cam_list = scene_data['translation_vec_cam']

    if 'mask_path' in config:
        mask_name_func = eval(config['mask_name_func'])
        mask_list = [os.path.join(config['mask_path'], mask_name_func(i)) for i in camera_labels]
    else:
        mask_list = [None] * len(target_list)

    if 'label_path' in config:
        label_name_func = eval(config['label_name_func'])
        label_list = [os.path.join(config['label_path'], label_name_func(i)) for i in camera_labels]
    else:
        label_list = [None] * len(target_list)

    assert hasattr(args, 'splitter_module') and hasattr(args, 'splitter_args')

    if len(config['val_indices']) > 0:
        splits = args.splitter_module([view_list, target_list, mask_list, label_list, rotation_matrix_ext_list, rotation_matrix_cam_list, translation_vec_ext_list, translation_vec_cam_list], val_indices)
    else:
        splits = args.splitter_module([view_list, target_list, mask_list, label_list, rotation_matrix_ext_list, rotation_matrix_cam_list, translation_vec_ext_list, translation_vec_cam_list], **args.splitter_args)

    view_list_train, view_list_val = splits[0]
    target_list_train, target_list_val = splits[1]
    mask_list_train, mask_list_val = splits[2]
    label_list_train, label_list_val = splits[3]
    rotation_matrix_ext_list_train, rotation_matrix_ext_list_val = splits[4]
    rotation_matrix_cam_list_train, rotation_matrix_cam_list_val = splits[5]
    translation_vec_ext_list_train, translation_vec_ext_list_val = splits[6]
    translation_vec_cam_list_train, translation_vec_cam_list_val = splits[7]

    avg_cam_center_train = np.mean(np.array(translation_vec_cam_list_train), axis=0, keepdims=True)
    dist = np.linalg.norm(np.array(translation_vec_cam_list_train) - avg_cam_center_train, axis=1, keepdims=True)
    radius = np.max(dist, axis=0)*1.1
    all_horizontal_vecs = (np.array(translation_vec_cam_list_train) - avg_cam_center_train)
    elev_angles = (np.degrees(np.arcsin(all_horizontal_vecs[:,1][...,np.newaxis]/dist)))
    sample_stddev_elev_angles = 1.5*np.std(elev_angles)

    radius = np.max(dist, axis=0)*1.1
    radius = radius[0]
    phi = np.linspace(0., 2. * np.pi, len(translation_vec_cam_list_train), endpoint=False)
    theta = np.linspace(-sample_stddev_elev_angles, sample_stddev_elev_angles, len(translation_vec_cam_list_train), endpoint=False)
    np.random.shuffle(phi)
    np.random.shuffle(theta)

    size_imgs = load_image(target_list_train[0]).shape
    new_crop_size = (16*(size_imgs[1]//16), 16*(size_imgs[0]//16))
    args.crop_size = new_crop_size

    ds_train = DynamicDataset(scene_data, args.crop_size, view_list_train, target_list_train, mask_list_train, label_list_train, rotation_matrix_ext_list_train, 
        rotation_matrix_cam_list_train, translation_vec_ext_list_train, translation_vec_cam_list_train, theta=theta, phi=phi, center=avg_cam_center_train, radius=radius, supersampling=args.supersampling, **args.train_dataset_args)

    ds_val = DynamicDataset(scene_data, args.crop_size, view_list_val, target_list_val, mask_list_val, label_list_val, rotation_matrix_ext_list_val,
        rotation_matrix_cam_list_val, translation_vec_ext_list_val, translation_vec_cam_list_val, supersampling=args.supersampling, **args.val_dataset_args)

    return ds_train, ds_val
