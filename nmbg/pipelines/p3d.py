import os, sys
from pathlib import Path

from torch import autograd, optim

from nmbg.pipelines import Pipeline
from nmbg.datasets.dynamic import get_datasets
from nmbg.models.texture import PointTexture
from nmbg.models.unet import UNet
from nmbg.models.conv import DCDiscriminator
from nmbg.models.compose import NetAndTexture
from nmbg.criterions.vgg_loss import VGGLoss
from nmbg.utils.train import to_device, set_requires_grad, save_model, unwrap_model, image_grid, to_numpy, load_model_checkpoint, freeze
from nmbg.utils.perform import TicToc, AccumDict, Tee
import numpy as np
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as run_checkpoint
import os

TextureOptimizerClass = optim.RMSprop

def get_discriminator(img_size):
    dcgan = DCDiscriminator(img_size=img_size)

    return dcgan

def get_net(input_channels, args):
    net = UNet(
        num_input_channels=input_channels, 
        num_output_channels=3, 
        feature_scale=args.net_size, 
        more_layers=0, 
        upsample_mode='bilinear', 
        norm_layer='bn', 
        last_act='', 
        conv_block=args.conv_block
        )

    return net

def get_texture(num_channels, size, args):
    if not hasattr(args, 'reg_weight'):
        args.reg_weight = 0.

    texture = PointTexture(num_channels, size, activation=args.texture_activation, reg_weight=args.reg_weight)

    if args.texture_ckpt:
        texture = load_model_checkpoint(args.texture_ckpt, texture)

    return texture


def backward_compat(args):
    if not hasattr(args, 'input_channels'):
        args.input_channels = None
    if not hasattr(args, 'conv_block'):
        args.conv_block = 'gated'

    if args.pipeline == 'npbg.pipelines.ogl.Pix2PixPipeline':
        if not hasattr(args, 'input_modality'):
            args.input_modality = 1

    return args


class TexturePipeline(Pipeline):
    def export_args(self, parser):
        parser.add_argument('--descriptor_size', type=int, default=72)
        parser.add_argument('--texture_size', type=int)
        parser.add_argument('--texture_ckpt', type=Path)
        parser.add('--texture_lr', type=float, default=1e-1)
        parser.add('--texture_activation', type=str, default='none')
        parser.add('--n_points', type=int, default=0, help='this is for inference')

    def create(self, args):
        args = backward_compat(args)

        if not args.input_channels:
            args.input_channels = [2*8] * args.num_mipmap

        net = get_net(args.input_channels, args)
        dcgan = get_discriminator(img_size=256) #get_discriminator(img_size=512) for single scene trainings

        textures_fg = {}
        textures_bg = {}
        point_clouds_fg = {}
        point_clouds_bg = {}
        faces_fg = {}
        faces_bg = {}

        if args.inference:
            if args.use_mesh:
                size = args.texture_size
            else:
                size = args.n_points
            textures = {
                0: get_texture(args.descriptor_size, size, args)
                }
        else:
            self.ds_train, self.ds_val, self.ds_name = get_datasets(args)

            if args.continue_epoch != 0:
                for ds in self.ds_train:
                    if args.use_mesh:
                        assert args.texture_size, 'set texture size'
                        size = args.texture_size
                    else:
                        assert ds.scene_data['mesh_fg'] is not None, 'set pointcloud fg'
                        assert ds.scene_data['mesh_bg'] is not None, 'set pointcloud bg'
                        extra_points = ds.scene_data['extra_points']
                        size_fg = ds.scene_data['mesh_fg']['xyz'].shape[0] 
                        size_bg = ds.scene_data['mesh_bg']['xyz'].shape[0] 
                    args.texture_ckpt = args.texture_folder + 'PointTexture_stage_0_epoch_' + str(args.continue_epoch-1) + '_' + self.ds_name[ds.id] + '_fg' + '.pth'
                    textures_fg[ds.id] = get_texture(args.descriptor_size, size_fg, args)
                    args.texture_ckpt = args.texture_folder + 'PointTexture_stage_0_epoch_' + str(args.continue_epoch-1) + '_' + self.ds_name[ds.id] + '_bg' + '.pth'
                    textures_bg[ds.id] = get_texture(args.descriptor_size, size_bg, args)
                    full_pc_fg = np.array(ds.scene_data['mesh_fg']['xyz'])
                    full_pc_bg = np.array(ds.scene_data['mesh_bg']['xyz'])
                    point_clouds_fg[ds.id] = ds.scene_data['mesh_fg']['xyz'] 
                    point_clouds_bg[ds.id] = ds.scene_data['mesh_bg']['xyz']
                    faces_fg[ds.id] = ds.scene_data['mesh_fg']['faces']
                    faces_bg[ds.id] = ds.scene_data['mesh_bg']['faces']

            else:
                for ds in self.ds_train:
                    if args.use_mesh:
                        assert args.texture_size, 'set texture size'
                        size = args.texture_size
                    else:
                        assert ds.scene_data['mesh_fg'] is not None, 'set pointcloud fg'
                        assert ds.scene_data['mesh_bg'] is not None, 'set pointcloud bg'
                        size_fg = ds.scene_data['mesh_fg']['xyz'].shape[0] 
                        size_bg = ds.scene_data['mesh_bg']['xyz'].shape[0] 
                    textures_fg[ds.id] = get_texture(args.descriptor_size, size_fg, args)
                    textures_bg[ds.id] = get_texture(args.descriptor_size, size_bg, args)
                    full_pc_fg = np.array(ds.scene_data['mesh_fg']['xyz'])
                    full_pc_bg = np.array(ds.scene_data['mesh_bg']['xyz'])
                    point_clouds_fg[ds.id] = ds.scene_data['mesh_fg']['xyz'] 
                    point_clouds_bg[ds.id] = ds.scene_data['mesh_bg']['xyz']
                    faces_fg[ds.id] = ds.scene_data['mesh_fg']['faces']
                    faces_bg[ds.id] = ds.scene_data['mesh_bg']['faces']

            self.optimizer1 = optim.Adam(net.parameters(), lr=args.lr)
            self.optimizer2 = optim.Adam(dcgan.parameters(), lr=args.lr)

            if len(textures_fg) == 1 and len(textures_bg) == 1:
                self._extra_optimizer_fg = TextureOptimizerClass(textures_fg[0].parameters(), lr=args.texture_lr)
                self._extra_optimizer_bg = TextureOptimizerClass(textures_bg[0].parameters(), lr=args.texture_lr)
            else:
                self._extra_optimizer_fg = None
                self._extra_optimizer_bg = None

            self.criterion = args.criterion_module(**args.criterion_args).cuda()

        ss = args.supersampling if hasattr(args, 'supersampling') else 1

        self.net = net
        self.dcgan = dcgan
        self.textures_fg = textures_fg
        self.textures_bg = textures_bg
        self.model = NetAndTexture(net, dcgan, textures_fg, textures_bg, point_clouds_fg, point_clouds_bg, faces_fg, faces_bg, ss, args.crop_size)

        self.args = args

    def state_objects(self):
        datasets = self.ds_train

        objs = {'net': self.net, 'dcgan': self.dcgan}
        objs.update({ds.name+'_fg': self.textures_fg[ds.id] for ds in datasets})
        objs.update({ds.name+'_bg': self.textures_bg[ds.id] for ds in datasets})

        return objs

    def dataset_load(self, dataset):
        self.model.load_textures([ds.id for ds in dataset])
        
        for ds in dataset:
            ds.load()


    def extra_optimizer(self, dataset):
        # if we have single dataset, don't recreate optimizer
        if self._extra_optimizer_fg is not None and self._extra_optimizer_bg is not None:
            lr_drop = self.optimizer1.param_groups[0]['lr'] / self.args.lr
            self._extra_optimizer_fg.param_groups[0]['lr'] = self.args.texture_lr * lr_drop
            self._extra_optimizer_bg.param_groups[0]['lr'] = self.args.texture_lr * lr_drop
            return self._extra_optimizer_fg, self._extra_optimizer_bg

        param_group_fg = []
        param_group_bg = []
        for ds in dataset:
            param_group_fg.append(
                {'params': self.textures_fg[ds.id].parameters()}
            )
            param_group_bg.append(
                {'params': self.textures_bg[ds.id].parameters()}
            )

        lr_drop = self.optimizer1.param_groups[0]['lr'] / self.args.lr

        return TextureOptimizerClass(param_group_fg, lr=self.args.texture_lr * lr_drop), TextureOptimizerClass(param_group_bg, lr=self.args.texture_lr * lr_drop)

    def dataset_unload(self, dataset):
        self.model.unload_textures()

        for ds in dataset:
            ds.unload()
            self.textures_fg[ds.id].null_grad()
            self.textures_bg[ds.id].null_grad()

    def get_net(self):
        return self.net

    def get_dcgan(self):
        return self.dcgan