import numpy as np
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
from torchvision import transforms
from nmbg.gl.utils import get_intrinsics_batched, get_ndc_f_c_batched, compute_bce
import torch.nn.functional as F
from nmbg.models.sh import eval_sh
from contextlib import nullcontext

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    PerspectiveCameras, 
    PointsRasterizationSettings,
    RasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    MeshRasterizer,
    AlphaCompositor,
    NormWeightedCompositor,
    TexturesVertex
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0: 
            idx = len(self) + idx

        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class ModelAndLoss(nn.Module):
    def __init__(self, model, loss, use_mask=False):
        super().__init__()
        self.model = model
        self.loss = loss
        self.use_mask = use_mask
        self.bce_loss = lambda x, y : compute_bce(x, y)

    def forward(self, *args, **kwargs):
        scene_id = args[0]
        ext_rot_mat = args[1]
        cam_rot_mat = args[2]
        ext_trans_vec = args[3]
        cam_trans_vec = args[4]
        intrinsic_matrix = args[5]
        intrinsic_matrix_pytorch3d = args[6]
        ray_directions_img = args[7]
        phase = args[8]
        target = args[-3]
        aug_flag = args[-2]
        gan_flag = args[-1]

        if not isinstance(scene_id, (tuple, list)):
            scene_id = [scene_id]

        if not isinstance(ext_rot_mat, (tuple, list)):
            ext_rot_mat = [ext_rot_mat]

        if not isinstance(cam_rot_mat, (tuple, list)):
            cam_rot_mat = [cam_rot_mat]

        if not isinstance(ext_trans_vec, (tuple, list)):
            ext_trans_vec = [ext_trans_vec]

        if not isinstance(cam_trans_vec, (tuple, list)):
            cam_trans_vec = [cam_trans_vec]

        if not isinstance(intrinsic_matrix_pytorch3d, (tuple, list)):
            intrinsic_matrix_pytorch3d = [intrinsic_matrix_pytorch3d]

        if not isinstance(intrinsic_matrix, (tuple, list)):
            intrinsic_matrix = [intrinsic_matrix]

        if not isinstance(ray_directions_img, (tuple, list)):
            ray_directions_img = [ray_directions_img]

        if not isinstance(phase, (tuple, list)):
            phase = [phase]

        if not isinstance(target, (tuple, list)):
            target_input = [target]

        if not isinstance(aug_flag, (tuple, list)):
            aug_flag_input = [aug_flag]

        if not isinstance(gan_flag, (tuple, list)):
            gan_flag_input = [gan_flag]

        output, output_new, target_new = self.model(*scene_id, *ext_rot_mat, *cam_rot_mat, *ext_trans_vec, *cam_trans_vec, *intrinsic_matrix, *intrinsic_matrix_pytorch3d, ray_directions_img, *phase, *target_input, *gan_flag_input, **kwargs)
        aug_flag = aug_flag.cpu().numpy()[0]

        if self.use_mask and 'mask' in kwargs and kwargs['mask'] is not None:
            if aug_flag == 0:
                loss1 = self.loss(output * kwargs['mask'], target)
            else:
                loss1 = 0
        else:
            if aug_flag == 0:
                if gan_flag == 0:
                    loss1 = self.loss(output, target) 
                    generator_loss = self.bce_loss(output_new, 1)
                    discriminator_loss = torch.tensor(0).to(device)
                else:
                    loss1 = torch.tensor(0).to(device)
                    generator_loss = torch.tensor(0).to(device)
                    discriminator_loss = self.bce_loss(target_new, 1) + self.bce_loss(output_new, 0)
            else:
                if gan_flag == 0:
                    loss1 = torch.tensor(0).to(device)
                    generator_loss = self.bce_loss(output_new, 1)
                    discriminator_loss = torch.tensor(0).to(device)
                else:
                    loss1 = torch.tensor(0).to(device)
                    generator_loss = torch.tensor(0).to(device)
                    discriminator_loss = self.bce_loss(target_new, 1) + self.bce_loss(output_new, 0)
        return output, loss1, generator_loss, discriminator_loss


class BoxFilter(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.seq = nn.Sequential(
            nn.ReflectionPad2d(kernel_size//2), 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=8)
        )

        self.weights_init(kernel_size)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, kernel_size):
        kernel = torch.ones((kernel_size, kernel_size)) / kernel_size ** 2
        self.seq[1].weight.data.copy_(kernel)


class GaussianLayer(nn.Module):
    _instance = None

    def __init__(self, in_channels, out_channels, kernel_size=21, sigma=3):
        super(GaussianLayer, self).__init__()
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(kernel_size//2), 
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, bias=None, groups=8)
        )

        self.weights_init(kernel_size, sigma)

    def forward(self, x):
        return self.seq(x)

    def weights_init(self, kernel_size, sigma):
        n= np.zeros((kernel_size, kernel_size))
        n[kernel_size//2, kernel_size//2] = 1
        k = gaussian_filter(n,sigma=sigma)
        for name, f in self.named_parameters():
            f.data.copy_(torch.from_numpy(k))

    @staticmethod
    def get_instance():
        if GaussianLayer._instance is None:
            GaussianLayer._instance = GaussianLayer(8, 8, kernel_size=13, sigma=6).cuda()

        return GaussianLayer._instance

class NetAndTexture(nn.Module):
    def __init__(self, net, dcgan, textures_fg, textures_bg, point_clouds_fg, point_clouds_bg, faces_fg, faces_bg, supersampling=1, crop_size=(512,512), temporal_average=False):
        super().__init__()
        
        self.net = net
        self.dcgan = dcgan
        self.ss = supersampling
        point_clouds_fg = dict(point_clouds_fg)
        point_clouds_bg = dict(point_clouds_bg)
        faces_fg = dict(faces_fg)
        faces_bg = dict(faces_bg)
        self._textures_fg = {k: v.cpu() for k, v in textures_fg.items()}
        self._textures_bg = {k: v.cpu() for k, v in textures_bg.items()}
        self._point_clouds_fg = {k: v for k, v in point_clouds_fg.items()}
        self._point_clouds_bg = {k: v for k, v in point_clouds_bg.items()}
        self._faces_fg = {k: v for k, v in faces_fg.items()}
        self._faces_bg = {k: v for k, v in faces_bg.items()}
        self._loaded_textures = []

        self.last_input = None
        self.temporal_average = temporal_average
        self.crop_size = crop_size
        self.sh_deg = 2

    def load_textures(self, texture_ids):
        if torch.is_tensor(texture_ids):
            texture_ids = texture_ids.cpu().tolist()
        elif isinstance(texture_ids, int):
            texture_ids = [texture_ids]

        for tid in texture_ids:
            self._modules[str(tid)+'_fg'] = self._textures_fg[tid]
            self._modules[str(tid)+'_bg'] = self._textures_bg[tid]

        self._loaded_textures = texture_ids

    def unload_textures(self):
        for tid in self._loaded_textures:
            self._modules[str(tid)+'_fg'].cpu()
            self._modules[str(tid)+'_bg'].cpu()
            del self._modules[str(tid)+'_fg']
            del self._modules[str(tid)+'_bg']

    def reg_loss(self):
        loss = 0
        for tid in self._loaded_textures:
            loss += self._modules[str(tid)+'_fg'].reg_loss()
            loss += self._modules[str(tid)+'_bg'].reg_loss()

        return loss

    def rescale_K(self, K_, sx, sy, keep_fov=False):
        K = K_.clone()
        K[:, 0, 2] = sx * K[:, 0, 2]
        K[:, 1, 2] = sy * K[:, 1, 2]
        if keep_fov:
            K[:, 0, 0] = sx * K[:, 0, 0]
            K[:, 1, 1] = sy * K[:, 1, 1]
        return K

    def renderer(self, rasterizer1, rasterizer2, point_clouds, mesh, ray_directions, camtransvec, res):
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(torch.abs(raw))*dists)
        fragments1 = rasterizer1(point_clouds)
        fragments2 = rasterizer2(mesh)
        verts_colors = point_clouds.features_packed()
        faces = mesh.faces_packed() 
        face_colors = verts_colors[faces]
        colors = interpolate_face_attributes(fragments2.pix_to_face, fragments2.bary_coords, face_colors)
        ray_dirns = ray_directions
        ray_dirns = torch.flip(ray_dirns, [1])
        ray_dirns = torch.flip(ray_dirns, [2])
        ray_dirns_new = ray_dirns.permute(1,2,0)
        ray_dirns = ray_dirns.unsqueeze(0)
        indices_new1_fg = fragments1.idx.long()[0]
        indices_new2_fg = fragments2.pix_to_face.long()[0]
        indices_new1_bg = fragments1.idx.long()[1]
        indices_new2_bg = fragments2.pix_to_face.long()[1]
        mask_indices1_fg = ~(indices_new1_fg==-1)
        mask_indices2_fg = ~(indices_new2_fg==-1)
        mask_indices1_bg = ~(indices_new1_bg==-1)
        mask_indices2_bg = ~(indices_new2_bg==-1)
        final_mask_fg = torch.logical_or(mask_indices1_fg, mask_indices2_fg)
        final_mask_bg = torch.logical_or(mask_indices1_bg, mask_indices2_bg)
        r = rasterizer1.raster_settings.radius

        dists2 = fragments1.dists
        weights = (1 - dists2 / (r * r))
        weights = weights.float()
        pc_feats_nerf1_fg = point_clouds.features_packed()[indices_new1_fg]
        pc_feats_nerf1_bg = point_clouds.features_packed()[indices_new1_bg]
        pc_feats_nerf1_fg = pc_feats_nerf1_fg*mask_indices1_fg.unsqueeze(-1)
        pc_feats_nerf1_bg = pc_feats_nerf1_bg*mask_indices1_bg.unsqueeze(-1)
        pc_feats_nerf_original1_fg = pc_feats_nerf1_fg.squeeze(2)
        pc_feats_nerf_original1_bg = pc_feats_nerf1_bg.squeeze(2)
        pc_feats_nerf_original1_fg = pc_feats_nerf_original1_fg*weights[0]
        pc_feats_nerf_original1_bg = pc_feats_nerf_original1_bg*weights[1]
        pc_feats_nerf_original2_fg = colors[0].squeeze(2)
        pc_feats_nerf_original2_bg = colors[1].squeeze(2)
        pc_feats_nerf_original_fg = torch.cat((pc_feats_nerf_original1_fg, pc_feats_nerf_original2_fg), axis=-1)
        pc_feats_nerf_original_bg = torch.cat((pc_feats_nerf_original1_bg, pc_feats_nerf_original2_bg), axis=-1)
        pc_feats_nerf_original_reshaped_fg = pc_feats_nerf_original_fg.view(1, -1, 144)
        pc_feats_nerf_original_reshaped_bg = pc_feats_nerf_original_bg.view(1, -1, 144)
        pc_feats_nerf_original_reshaped_fg = pc_feats_nerf_original_reshaped_fg.squeeze(0)
        pc_feats_nerf_original_reshaped_bg = pc_feats_nerf_original_reshaped_bg.squeeze(0)

        ray_dirns_sh = ray_dirns[0].permute(1,2,0).view(-1,3)
        wer_fg = pc_feats_nerf_original_reshaped_fg.reshape(*pc_feats_nerf_original_reshaped_fg.shape[:-1], -1, (self.sh_deg + 1) ** 2)
        wer_bg = pc_feats_nerf_original_reshaped_bg.reshape(*pc_feats_nerf_original_reshaped_bg.shape[:-1], -1, (self.sh_deg + 1) ** 2)
        out_feat_fg = eval_sh(2, wer_fg, ray_dirns_sh)
        out_feat_bg = eval_sh(2, wer_bg, ray_dirns_sh)
        out_feat_fg = out_feat_fg.unsqueeze(0)
        out_feat_bg = out_feat_bg.unsqueeze(0)
        final_mask_fg = final_mask_fg.unsqueeze(0)
        final_mask_bg = final_mask_bg.unsqueeze(0)
        out_feat_fg = out_feat_fg.view(1, indices_new1_fg.shape[0], indices_new1_fg.shape[1], 16)
        out_feat_bg = out_feat_bg.view(1, indices_new1_bg.shape[0], indices_new1_bg.shape[1], 16)
        return out_feat_fg, out_feat_bg, final_mask_fg, final_mask_bg

    def forward(self, scene_id, ext_rot_mat, cam_rot_mat, ext_trans_vec, cam_trans_vec, intrinsic_matrix, intrinsic_matrix_pytorch3d, ray_directions_img, phase, target_input, gan_flag, **kwargs):
        if gan_flag == 0:
            cm = nullcontext()
        else:
            cm = torch.no_grad()
        with cm:
            out = []
            mask_fg = []
            mask_bg = []
            nerf_img_output = []
            up_16 = []
            in_16 = []

            batch_size = ext_rot_mat.shape[0]
            scene_ids = scene_id['id']

            focal_length, p_x, p_y = get_intrinsics_batched(intrinsic_matrix_pytorch3d)

            if torch.is_tensor(scene_ids):
                scene_ids = scene_ids.tolist()
            elif isinstance(scene_ids, int):
                scene_ids = [scene_ids]

            j = 0
            for i, tid in enumerate(scene_ids): # per item in batch
                texture_fg = self._modules[str(tid)+'_fg'].texture_.squeeze(0).transpose(0,1).float()
                texture_bg = self._modules[str(tid)+'_bg'].texture_.squeeze(0).transpose(0,1).float()
                pcloud_fg = self._point_clouds_fg[tid]
                pcloud_bg = self._point_clouds_bg[tid]
                faces_fg = self._faces_fg[tid]
                faces_bg = self._faces_bg[tid]
                verts_fg = torch.Tensor(pcloud_fg).to(device).float()
                verts_bg = torch.Tensor(pcloud_bg).to(device).float()
                faces_fg = faces_fg.to(device)
                faces_bg = faces_bg.to(device)
                rotmat = ext_rot_mat[j].transpose(0,1).unsqueeze(0)
                transvec = ext_trans_vec[j].unsqueeze(0)
                rotmatext = ext_rot_mat[j]
                transvecext = ext_trans_vec[j]
                camrotmat = cam_rot_mat[j]
                camtransvec = cam_trans_vec[j]
                ray_directions = ray_directions_img    
                point_clouds = Pointclouds(points=[verts_fg, verts_bg], features=[texture_fg, texture_bg])    
                meshes = Meshes(verts=[verts_fg, verts_bg], faces=[faces_fg, faces_bg]) 
                focal_length_idx = focal_length[j]   
                p_x_idx = p_x[j]
                p_y_idx = p_y[j] 
                j += 1       

                input_multiscale_fg = []
                input_multiscale_bg = []
                mask_multiscale_fg = []
                mask_multiscale_bg = []
                nerf_img_multiscale = []
                for k in range(5):
                    tex_sample = None
                    input_ex_fg = []
                    input_ex_bg = []
                    mask_ex_fg = []
                    mask_ex_bg = []
                    nerf_img_ex = []
                    vs = self.ss * target_input.shape[2] // 2 ** k, self.ss * target_input.shape[3] // 2 ** k
                
                    res_i_0 = (self.crop_size[1] // 2 ** k)
                    res_i_1 = (self.crop_size[0] // 2 ** k) 
                    mx = j - 1
                    ray_directions_res = ray_directions[k][mx]

                    focal_length_idx_new = focal_length_idx / (2 ** k)
                    p_x_idx_new = p_x_idx / (2 ** k)
                    p_y_idx_new = p_y_idx / (2 ** k)

                    raster_settings1 = PointsRasterizationSettings(
                        image_size=(vs[0], vs[1]), 
                        radius = 0.006*(2 ** k),
                        points_per_pixel = 1,
                        max_points_per_bin = 400000
                    )

                    raster_settings2 = RasterizationSettings(
                        image_size=(vs[0], vs[1]), 
                        blur_radius=0.0, 
                        faces_per_pixel=1, 
                    )
                    
                    f_ndc, px_ndc, py_ndc = get_ndc_f_c_batched(focal_length_idx_new, p_x_idx_new, p_y_idx_new, vs[1], vs[0]) 
                    f_ndc = f_ndc.float()
                    px_ndc = px_ndc.float()
                    py_ndc = py_ndc.float()
                    f_ndc = (f_ndc,)
                    prp_ndc = ((px_ndc, py_ndc), )
                    rotmat = rotmat.float()
                    transvec = transvec.float()
                    cameras = PerspectiveCameras(device=device, focal_length=f_ndc, principal_point=prp_ndc, R=rotmat, T=transvec)

                    rasterizer1 = PointsRasterizer(cameras=cameras, raster_settings=raster_settings1)
                    rasterizer2 = MeshRasterizer(cameras=cameras, raster_settings=raster_settings2)

                    tex_sample_fg, tex_sample_bg, mask_fg, mask_bg = self.renderer(rasterizer1, rasterizer2, point_clouds, meshes, ray_directions_res, camtransvec, k)

                    tex_sample_fg = torch.flip(tex_sample_fg, [1])
                    tex_sample_fg = torch.flip(tex_sample_fg, [2])

                    tex_sample_bg = torch.flip(tex_sample_bg, [1])
                    tex_sample_bg = torch.flip(tex_sample_bg, [2])

                    mask_fg = torch.flip(mask_fg, [1])
                    mask_fg = torch.flip(mask_fg, [2])

                    mask_bg = torch.flip(mask_bg, [1])
                    mask_bg = torch.flip(mask_bg, [2])

                    tex_sample_fg = tex_sample_fg.transpose(2,3).transpose(1,2)
                    tex_sample_fg = tex_sample_fg.float()

                    tex_sample_bg = tex_sample_bg.transpose(2,3).transpose(1,2)
                    tex_sample_bg = tex_sample_bg.float()

                    mask_fg = mask_fg.transpose(2,3).transpose(1,2)
                    mask_bg = mask_bg.transpose(2,3).transpose(1,2)

                    input_fg_cat = torch.cat(input_ex_fg + [tex_sample_fg], 1)
                    input_bg_cat = torch.cat(input_ex_bg + [tex_sample_bg], 1)
                    mask_fg_cat = torch.cat(mask_ex_fg + [mask_fg], 1)
                    mask_bg_cat = torch.cat(mask_ex_bg + [mask_bg], 1)

                    if self.ss > 1:
                        input_cat = nn.functional.interpolate(input_cat, scale_factor=1./self.ss, mode='bilinear')
                    
                    input_multiscale_fg.append(input_fg_cat)
                    input_multiscale_bg.append(input_bg_cat)
                    mask_multiscale_fg.append(mask_fg_cat)
                    mask_multiscale_bg.append(mask_bg_cat)

                if self.temporal_average:
                    if self.last_input is not None:
                        for i in range(len(input_multiscale)):
                            input_multiscale[i] = (input_multiscale[i] + self.last_input[i]) / 2
                    self.last_input = list(input_multiscale)
                    
                input_multiscale_all = input_multiscale_fg + input_multiscale_bg
                out1 = self.net(*input_multiscale_all, **kwargs)
                out.append(out1)

            out = torch.cat(out, 0)

        out_new = F.interpolate(out, (256, 256), mode='bilinear')
        target_new = F.interpolate(target_input, (256, 256), mode='bilinear') 
        out_new.requires_grad_()
        target_new.requires_grad_()
        out_new = self.dcgan(out_new)
        target_new = self.dcgan(target_new)

        if kwargs.get('return_input'):
            return out, out_new, target_new, input_multiscale
        else:
            return out, out_new, target_new
