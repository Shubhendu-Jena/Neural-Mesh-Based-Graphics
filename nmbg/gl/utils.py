import os, sys
import cv2
import numpy as np
import pickle
import time
import yaml
import trimesh

import torch
import torch.nn.functional as F

import xml.etree.ElementTree as ET

from sklearn.decomposition import IncrementalPCA
import itertools
from pytorch3d.io import load_objs_as_meshes, load_obj, load_ply

def compute_bce(d_out, target):
    targets = d_out.new_full(size=d_out.size(), fill_value=target)
    loss = F.binary_cross_entropy_with_logits(d_out, targets)
    return loss

class TicToc:
    def __init__(self):
        self.tic_toc_tic = None

    def tic(self):
        self.tic_toc_tic = time.time()

    def toc(self):
        assert self.tic_toc_tic, 'You forgot to call tic()'
        return (time.time() - self.tic_toc_tic) * 1000

    def tocp(self, str):
        print(f"{str} took {self.toc():.4f}ms")

    @staticmethod
    def print_timing(timing, name=''):
        print(f'\n=== {name} Timimg ===')
        for fn, times in timing.items():
            min, max, mean, p95 = np.min(times), np.max(times), np.mean(times), np.percentile(times, 95)
            print(f'{fn}:\tmin: {min:.4f}\tmax: {max:.4f}\tmean: {mean:.4f}ms\tp95: {p95:.4f}ms')


class FastRand:
    def __init__(self, shape, tform, bank_size):
        bank = []
        for i in range(bank_size):
            p = np.random.rand(*shape)
            p = tform(p)
            bank.append(p)

        self.bank = bank

    def toss(self):
        i = np.random.randint(0, len(self.bank))
        return self.bank[i]


def cv2_write(fn, x):
    x = np.clip(x, 0, 1) * 255
    x = x[..., :3][..., ::-1]
    cv2.imwrite(fn, x.astype(np.uint8))


def to_numpy(x, float16=False, flipv=True):
    if not isinstance(x, np.ndarray):
        x = x.detach().cpu().numpy()

    if float16:
        x = x.astype(np.float16)

    if flipv:
        x = x[::-1].copy()

    return x


def pca_color(tex, save='', load=''):
    tex = tex[0].transpose(1, 0)
    if load and os.path.exists(load):
        print('loading...')
        with open(load,'rb') as f:
            pca=pickle.load(f)
        print('applying...')
        res=pca.transform(tex)
    else:
        pca=IncrementalPCA(n_components=3, batch_size=64)
        print('applying...')
        res=pca.fit_transform(tex)
    if save and save != load:
        with open(save,'wb') as f:
            pickle.dump(pca,f)
    # pca_color_n=(pca_color - pca_color.min()) / (pca_color.max() - pca_color.min())
    # return res.transpose(1, 0)[None]
    return res


def crop_proj_matrix(pm, old_w, old_h, new_w, new_h):
    # NOTE: this is not precise
    old_cx = old_w / 2
    old_cy = old_h / 2
    new_cx = new_w / 2
    new_cy = new_h / 2

    pm_new = pm.copy()
    pm_new[0,0] = pm[0,0]*old_w/new_w
    pm_new[0,2] = (pm[0,2]-1)*old_w*new_cx/old_cx/new_w + 1
    pm_new[1,1] = pm[1,1]*old_h/new_h
    pm_new[1,2] = (pm[0,2]+1)*old_h*new_cy/old_cy/new_h - 1
    return pm_new


def recalc_proj_matrix_planes(pm, new_near=.01, new_far=1000.):
    depth = float(new_far - new_near)
    q = -(new_far + new_near) / depth
    qn = -2 * (new_far * new_near) / depth

    out = pm.copy()

    # Override near and far planes
    out[2, 2] = q
    out[2, 3] = qn

    return out


def get_proj_matrix(K, image_size, znear=.01, zfar=1000.):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    width, height = image_size
    m = np.zeros((4, 4))
    m[0][0] = 2.0 * fx / width
    m[0][1] = 0.0
    m[0][2] = 0.0
    m[0][3] = 0.0

    m[1][0] = 0.0
    m[1][1] = 2.0 * fy / height
    m[1][2] = 0.0
    m[1][3] = 0.0

    m[2][0] = 1.0 - 2.0 * cx / width
    m[2][1] = 2.0 * cy / height - 1.0
    m[2][2] = (zfar + znear) / (znear - zfar)
    m[2][3] = -1.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 2.0 * zfar * znear / (znear - zfar)
    m[3][3] = 0.0

    return m.T

def get_proj_matrix_new(K):
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    m = np.zeros((4, 4))
    m[0][0] = fx
    m[0][1] = 0.0
    m[0][2] = cx
    m[0][3] = 0.0

    m[1][0] = 0.0
    m[1][1] = fy
    m[1][2] = cy
    m[1][3] = 0.0

    m[2][0] = 0
    m[2][1] = 0
    m[2][2] = 0
    m[2][3] = 1.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 1.0
    m[3][3] = 0.0

    return m


def get_ndc_f_c(f, px, py, image_width, image_height):
    #import ipdb; ipdb.set_trace()
    s = min(image_width, image_height)
    f_ndc = f * 2.0 / (s - 1)

    px_ndc = - (px - (image_width - 1) / 2.0) * 2.0 / (s - 1)
    py_ndc = - (py - (image_height - 1) / 2.0) * 2.0 / (s - 1)

    return f_ndc, px_ndc, py_ndc

def get_ndc_f_c_batched(f, px, py, image_width, image_height):
    s = min(image_width, image_height)
    f_ndc = f * 2.0 / (s - 1)

    px_ndc = - (px - (image_width - 1) / 2.0) * 2.0 / (s - 1)
    py_ndc = - (py - (image_height - 1) / 2.0) * 2.0 / (s - 1)

    return f_ndc, px_ndc, py_ndc

def get_intrinsics(K):
    f = K[0,0]
    cx = K[0,2]
    cy = K[1,2]
    return f, cx, cy

def get_intrinsics_batched(K):
    f = K[:, 0, 0]
    cx = K[:, 0, 2]
    cy = K[:, 1, 2]
    return f, cx, cy

def rescale_K(K_, sx, sy, keep_fov=True):
    K = K_.copy()
    K[0, 2] = sx * K[0, 2]
    K[1, 2] = sy * K[1, 2]
    if keep_fov:
        K[0, 0] = sx * K[0, 0]
        K[1, 1] = sy * K[1, 1]
    return K


def crop_intrinsic_matrix(K, old_size, new_size):
    K = K.copy()
    K[0, 2] = new_size[0] * K[0, 2] / old_size[0]
    K[1, 2] = new_size[1] * K[1, 2] / old_size[1]
    return K


def intrinsics_from_xml(xml_file):
    root = ET.parse(xml_file).getroot()
    calibration = root.find('chunk/sensors/sensor/calibration')
    resolution = calibration.find('resolution')
    width = float(resolution.get('width'))
    height = float(resolution.get('height'))
    f = float(calibration.find('f').text)
    cx = width/2
    cy = height/2

    K = np.array([
        [f, 0, cx],
        [0, f, cy],
        [0, 0,  1]
        ], dtype=np.float32)

    return K, (width, height)


def extrinsics_from_xml(xml_file, verbose=False):
    root = ET.parse(xml_file).getroot()
    transforms = {}
    for e in root.findall('chunk/cameras')[0].findall('camera'):
        label = e.get('label')
        try:
            transforms[label] = e.find('transform').text
        except:
            if verbose:
                print('failed to align camera', label)

    view_matrices = []
    # labels_sort = sorted(list(transforms), key=lambda x: int(x))
    labels_sort = list(transforms)
    for label in labels_sort:
        extrinsic = np.array([float(x) for x in transforms[label].split()]).reshape(4, 4)
        extrinsic[:, 1:3] *= -1
        view_matrices.append(extrinsic)

    return view_matrices, labels_sort


def extrinsics_from_view_matrix(path):
    vm = np.loadtxt(path).reshape(-1,4,4)
    vm, ids = get_valid_matrices(vm)

    # we want consistent camera label data type, as cameras from xml
    ids = [str(i) for i in ids]

    return vm, ids

def load_scene_data(path):
    im_ext = ".jpg"
    
    rot_ext = str(path / "Rs.npy")
    rot_mat = np.load(rot_ext)
    rot_mat_ext = np.copy(rot_mat)

    rot_mat_ext_new = []
    for idx in range(rot_mat_ext.shape[0]):
        rot_mat_ext_new.append(rot_mat_ext[idx])

    rot_cam = np.zeros_like(rot_mat)

    for idx in range(rot_mat.shape[0]):
        rot_cam[idx] = np.transpose(rot_mat[idx]) 

    trans_ext = str(path / "ts.npy")
    trans_mat = np.load(trans_ext)
    trans_vec_ext = np.copy(trans_mat)

    trans_vec_ext_new = []
    for idx in range(trans_vec_ext.shape[0]):
        trans_vec_ext_new.append(trans_vec_ext[idx])

    trans_cam = np.zeros_like(trans_mat)
    for idx in range(rot_cam.shape[0]):
        trans_cam[idx, :] = np.matmul(-rot_cam[idx], trans_mat[idx,:]) 

    trans_vec_cam_new = []
    for idx in range(trans_cam.shape[0]):
        trans_vec_cam_new.append(trans_cam[idx])

    rot_mat_cam = np.copy(rot_cam)
    rot_mat_cam_new = []
    for idx in range(rot_mat_cam.shape[0]):
        rot_mat_cam_new.append(rot_mat_cam[idx])

    rot_mat = rot_cam
    trans_mat = trans_cam
    trans_mat = trans_mat[...,np.newaxis]
    view_matrix = np.concatenate((rot_mat, trans_mat), axis=2)
    view_matrix = np.insert(view_matrix, 3, [0,0,0,1], axis=1)

    view_matrix_new = []
    for idx in range(view_matrix.shape[0]):
        view_matrix_new.append(view_matrix[idx])

    view_matrix = view_matrix_new

    pcloud_fg = str(path / ".." / "delaunay_photometric_fg_0.50.ply")
    pcloud_bg = str(path / ".." / "delaunay_photometric_bg_0.50.ply")
    uv_order = 's,t'
    mesh_fg = import_model3d(pcloud_fg)
    mesh_bg = import_model3d(pcloud_bg)
    samples = None
    pointcloud = None

    texture = None

    intrinsic_ext = str(path / "Ks.npy")
    len_scene = len(np.load(intrinsic_ext))
    intrinsic_matrix = np.load(intrinsic_ext)[0]
    intrinsic_matrix_inv = np.expand_dims(np.linalg.inv(intrinsic_matrix), axis=0)
    intrinsic_matrix_inv = np.repeat(intrinsic_matrix_inv, len_scene, axis=0)

    proj_matrix = None
    model3d_origin = np.eye(4)
    point_sizes = None

    net_ckpt = None
    tex_ckpt = None
    camera_labels = None
    config = None

    return {
    'pointcloud': pointcloud,
    'extra_points': samples,
    'point_sizes': point_sizes,
    'mesh_fg': mesh_fg,
    'mesh_bg': mesh_bg,
    # 'use_mesh': use_mesh,
    'texture': texture,
    'proj_matrix': proj_matrix,
    'intrinsic_matrix': intrinsic_matrix,
    'rotation_matrix_ext': rot_mat_ext_new,
    'rotation_matrix_cam': rot_mat_cam_new,
    'translation_vec_ext': trans_vec_ext_new,
    'translation_vec_cam': trans_vec_cam_new,
    'view_matrix': view_matrix,
    'camera_labels': camera_labels,
    'model3d_origin': model3d_origin,
    'config': config,

    'net_ckpt': net_ckpt,
    'tex_ckpt': tex_ckpt
    }

def fix_relative_path(path, config_path):
    if not os.path.exists(path) and not os.path.isabs(path):
        root = os.path.dirname(config_path)
        abspath = os.path.join(root, path)
        if os.path.exists(abspath):
            return abspath
    return path


def get_valid_matrices(mlist):
    ilist = []
    vmlist = []
    for i, m in enumerate(mlist):
        if np.isfinite(m).all():
            ilist.append(i)
            vmlist.append(m)

    return vmlist, ilist


def get_xyz_colors(xyz, r=8):
    mmin, mmax = xyz.min(axis=0), xyz.max(axis=0)
    color = (xyz - mmin) / (mmax - mmin)
    # color = 0.5 + 0.5 * xyz / r
    return np.clip(color, 0., 1.).astype(np.float32)

def get_normal_colors(normals):
    # [-1,+1]->[0,1]
    return (normals * 0.5 + 0.5).astype(np.float32)

def import_model3d(model_path):
    verts, faces = load_ply(model_path)

    n_pts = verts.shape[0]

    model = {
        'rgb': None,
        'normals': None,
        'uv2d': None,
        'faces': None
    }

    model['xyz'] = verts
    model['faces'] = faces

    print('=== 3D model ===')
    print('VERTICES: ', n_pts)
    print('EXTENT: ', model['xyz'].min(0), model['xyz'].max(0))
    print('================')

    return model

def get_vec(view_mat):
    view_mat = view_mat.copy()
    rvec0 = cv2.Rodrigues(view_mat[:3, :3])[0].flatten()
    t0 = view_mat[:3, 3]
    return rvec0, t0


def nearest_train(view_mat, test_pose, p=0.05):
    dists = []
    angs = []
    test_rvec, test_t = get_vec(test_pose)
    for i in range(len(view_mat)):
        rvec, t = get_vec(view_mat[i])
        dists.append(
            np.linalg.norm(test_t - t)
        )
        angs.append(
            np.linalg.norm(test_rvec - rvec)
        )
    angs_sort = np.argsort(angs)
    angs_sort = angs_sort[:int(len(angs_sort) * p)]
    dists_pick = [dists[i] for i in angs_sort]
    ang_dist_i = angs_sort[np.argmin(dists_pick)]
    return ang_dist_i #, angs_sort[0]

