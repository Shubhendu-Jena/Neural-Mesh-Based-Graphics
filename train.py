import numpy as np
import random
import cv2
import yaml
from time import time, sleep
from collections import defaultdict
from pprint import pprint
from pathlib import Path

import os, sys
import datetime
import argparse

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import torch.utils.data
from torchvision import transforms

from tensorboardX import SummaryWriter
import torch.nn.functional as F

from nmbg.utils.perform import TicToc, AccumDict, Tee
from nmbg.utils.arguments import MyArgumentParser, eval_args
from nmbg.models.compose import ModelAndLoss
from nmbg.utils.train import to_device, image_grid, to_numpy, get_module, freeze, load_model_checkpoint, unwrap_model
from nmbg.utils.metric import PSNRMetric, SSIMMetric
from nmbg.pipelines import save_pipeline
import gc
torch.cuda.set_device(0)

def setup_environment(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

    cv2.setNumThreads(0)
    cv2.ocl.setUseOpenCL(False)
    os.environ['OMP_NUM_THREADS'] = '1'


def setup_logging(save_dir):
    tee = Tee(os.path.join(save_dir, 'log.txt'))
    sys.stdout, sys.stderr = tee, tee



def get_experiment_name(args, default_args, args_to_ignore, delimiter='__'):
    s = []

    args = vars(args)
    default_args = vars(default_args)

    def shorten_paths(args):
        args = dict(args)
        for arg, val in args.items():
            if isinstance(val, Path):
                args[arg] = val.name
        return args

    args = shorten_paths(args)
    default_args = shorten_paths(default_args)

    for arg in sorted(args.keys()):
        if arg not in args_to_ignore and default_args[arg] != args[arg]:
            s += [f"{arg}^{args[arg]}"]
    
    out = delimiter.join(s)
    out = out.replace('/', '+')
    out = out.replace("'", '')
    out = out.replace("[", '')
    out = out.replace("]", '')
    out = out.replace(" ", '')
    return out


def make_experiment_dir(base_dir, postfix='', use_time=True):
    time = datetime.datetime.now()

    if use_time:
        postfix = time.strftime(f"%m-%d_%H-%M-%S___{postfix}")

    save_dir = os.path.join(base_dir, postfix)
    os.makedirs(f'{save_dir}/checkpoints', exist_ok=True)

    return save_dir


def num_param(model):
    return sum([p.numel() for p in unwrap_model(model).parameters()])


def run_epoch(pipeline, phase, epoch, args, iter_cb=None):
    ad = AccumDict()
    tt = TicToc()

    psnr = PSNRMetric()
    ssim = SSIMMetric()
    
    device = 'cuda:0'         

    model = pipeline.model
    criterion = pipeline.criterion
    optimizer1 = pipeline.optimizer1
    optimizer2 = pipeline.optimizer2

    print(f'model parameters: {num_param(model)}')

    if args.merge_loss:
        model = ModelAndLoss(model, criterion, use_mask=args.use_mask)

    if args.multigpu:
        model = nn.DataParallel(model)

    def run_sub(dl, extra_optimizer_fg, extra_optimizer_bg):
        model.cuda()

        tt.tic()
        for it, data in enumerate(dl):
            scene_id = to_device(data['scene_id'], device)
            aug_flag = to_device(data['augment_cams'], device)
            ray_directions_img = to_device(data['ray_directions_img'], device)
            ext_rot_mat = to_device(data['ext_rot_mat'], device)
            ext_trans_vec = to_device(data['ext_trans_vec'], device)
            cam_rot_mat = to_device(data['cam_rot_mat'], device)
            cam_trans_vec = to_device(data['cam_trans_vec'], device)
            intrinsic_matrix = to_device(data['intrinsic_matrix'], device)
            intrinsic_matrix_pytorch3d = to_device(data['intrinsic_matrix_pytorch3d'], device)
            target = to_device(data['target'], device)

            if 'mask' in data and args.use_mask:
                mask = to_device(data['mask'], device)

                if mask.sum() < 1:
                    print(f'skip batch, mask is {mask.sum()}')
                    continue

                target *= mask
            else:
                mask = None

            ad.add('data_time', tt.toc())

            if phase == 'train':
                run_nos = 2
            else:
                run_nos = 1

            tt.tic()
            gan_flag = -1
            for idx in range(run_nos):
                gan_flag = gan_flag + 1
                if args.merge_loss:
                    out, loss1, generator_loss, discriminator_loss = model(scene_id, ext_rot_mat, cam_rot_mat, ext_trans_vec, cam_trans_vec, intrinsic_matrix, intrinsic_matrix_pytorch3d, ray_directions_img, phase, target, aug_flag, gan_flag, mask=mask)
                    loss = loss1 + generator_loss + discriminator_loss
                else:
                    out = model(input)

                    if mask is not None and args.use_mask:
                        loss = criterion(out * mask, target)
                    else:
                        loss = criterion(out, target)

                if loss.numel() > 1:
                    loss = loss.mean()
                    loss1 = loss1.mean()
                    generator_loss = generator_loss.mean()
                    discriminator_loss = discriminator_loss.mean()

                if mask is not None:
                    loss /= mask.mean() + 1e-6

                    # TODO: parameterize
                    bkg_color = torch.FloatTensor([1, 1, 1]).reshape(1, 3, 1, 1).to(loss.device)
                    bkg_weight = 500

                    n_mask = 1 - mask
                    out_bkg = out * n_mask
                    bkg = bkg_color * n_mask
                    loss_bkg = bkg_weight * torch.abs((out_bkg - bkg)).mean() / (n_mask.mean() + 1e-6)

                    loss += loss_bkg

                    ad.add('loss_bkg', loss_bkg.item())

                if hasattr(pipeline.model, 'reg_loss'):
                    reg_loss = pipeline.model.reg_loss()
                    loss += reg_loss

                    if torch.is_tensor(reg_loss):
                        reg_loss = reg_loss.item()
                    ad.add('reg_loss', reg_loss)

                ad.add('batch_time', tt.toc())
                if phase == 'train':
                    tt.tic()
                    loss.backward(create_graph=False)
                    
                    if gan_flag == 0:
                        optimizer1.step()
                        optimizer1.zero_grad()

                        if extra_optimizer_fg is not None:
                            extra_optimizer_fg.step()
                            extra_optimizer_fg.zero_grad()
                        if extra_optimizer_bg is not None:
                            extra_optimizer_bg.step()
                            extra_optimizer_bg.zero_grad()

                    else:
                        optimizer2.step()
                        optimizer2.zero_grad()
                    
                    ad.add('step_time_'+str(idx), tt.toc())
                    ad.add('loss1_'+str(idx), loss1.item())
                    ad.add('loss_'+str(idx), loss.item())
                    ad.add('generator_loss_'+str(idx), generator_loss.item())
                    ad.add('discriminator_loss_'+str(idx), discriminator_loss.item())

                else:
                    ad.add('loss1_'+str(idx), loss1.item())
                    ad.add('loss_'+str(idx), loss.item())

                    out_metric = out.transpose(1,2).transpose(2,3).detach().cpu().numpy()
                    target_metric = target.transpose(1,2).transpose(2,3).detach().cpu().numpy()

                    ad.add('loss1PSNR', np.mean(psnr.add(out_metric, target_metric)))
                    ad.add('loss1SSIM', np.mean(ssim.add(out_metric, target_metric)))

                if iter_cb:
                    tt.tic()
                    iter_cb.on_iter(it + it_before, max_it, input, out, target, data, ad, str(idx), phase, epoch)
            
            tt.tic() # data_time

    ds_list = pipeline.__dict__[f'ds_{phase}']


    sub_size = args.max_ds


    if phase == 'train':
        random.shuffle(ds_list)
    #     random.shuffle(ds_aug_list)

    it_before = 0
    max_it = np.sum([len(ds) for ds in ds_list]) // args.batch_size

    for i_sub in range(0, len(ds_list), sub_size):
        ds_sub = ds_list[i_sub:i_sub + sub_size]
        ds_ids = [d.id for d in ds_sub]
        print(f'running on datasets {ds_ids}')

        ds = ConcatDataset(ds_sub)
        if phase == 'train':
            dl = DataLoader(ds, args.batch_size, num_workers=args.dataloader_workers, drop_last=True, pin_memory=True, shuffle=True, worker_init_fn=ds_init_fn)
        else:
            batch_size_val = args.batch_size if args.batch_size_val is None else args.batch_size_val
            dl = DataLoader(ds, batch_size_val, num_workers=args.dataloader_workers, drop_last=True, pin_memory=True, shuffle=False, worker_init_fn=ds_init_fn)

        pipeline.dataset_load(ds_sub)
        print(f'total parameters: {num_param(model)}')

        extra_optimizer_fg, extra_optimizer_bg = pipeline.extra_optimizer(ds_sub)

        run_sub(dl, extra_optimizer_fg, extra_optimizer_bg)

        pipeline.dataset_unload(ds_sub)

        it_before += len(dl)

        torch.cuda.empty_cache()

    avg_loss = np.mean(ad['loss_0'])
    iter_cb.on_epoch(phase, avg_loss, epoch)
    avg_psnr1 = np.mean(ad['loss1PSNR'])
    avg_ssim1 = np.mean(ad['loss1SSIM'])
    iter_cb.on_epoch(phase, avg_psnr1, epoch)
    iter_cb.on_epoch(phase, avg_ssim1, epoch)

    return avg_loss, avg_psnr1, avg_ssim1


def run_train(epoch, pipeline, args, iter_cb):

    if args.eval_in_train or (args.eval_in_train_epoch >= 0 and epoch >= args.eval_in_train_epoch):
        print('EVAL MODE IN TRAIN')
        pipeline.model.eval()
        if hasattr(pipeline.model, 'ray_block') and pipeline.model.ray_block is not None:
            pipeline.model.ray_block.train()
    else:
        pipeline.model.train()

    with torch.set_grad_enabled(True):
        return run_epoch(pipeline, 'train', epoch, args, iter_cb=iter_cb)


def run_eval(epoch, pipeline, args, iter_cb):
    torch.cuda.empty_cache()

    if args.eval_in_test:
        pipeline.model.eval()
    else:
        print('TRAIN MODE IN EVAL')
        pipeline.model.train()

    with torch.set_grad_enabled(False):
        return run_epoch(pipeline, 'val', epoch, args, iter_cb=iter_cb)
    

class TrainIterCb:
    def __init__(self, args, writer):
        self.args = args
        self.writer = writer
        self.train_it = 0

    def on_iter(self, it, max_it, input, out, target, data_dict, ad, idx, phase, epoch):  
        if it % self.args.log_freq == 0:
            s = f'{phase.capitalize()}: [{epoch}][{it}/{idx}/{max_it-1}]\t'
            s += str(ad)
            print(s)

        if phase == 'train':
            self.writer.add_scalar(f'{phase}/loss_'+idx, ad['loss_'+idx][-1], self.train_it)
            self.writer.add_scalar(f'{phase}/generator_loss_'+idx, ad['generator_loss_'+idx][-1], self.train_it)
            self.writer.add_scalar(f'{phase}/discriminator_loss_'+idx, ad['discriminator_loss_'+idx][-1], self.train_it)

            if 'reg_loss' in ad.__dict__():
                self.writer.add_scalar(f'{phase}/reg_loss', ad['reg_loss'][-1], self.train_it)

            self.train_it += 1

        if it % self.args.log_freq_images == 0:
            if isinstance(out, dict):
                inputs = out['input']
                scale = np.random.choice(len(inputs))
                keys = list(inputs.keys())
                out_img = inputs[keys[scale]]
                out = F.interpolate(out_img, size=target.shape[2:])
            
            out = out.clamp(0, 1)
            self.writer.add_image(f'{phase}', image_grid(out, target), self.train_it)

    def on_epoch(self, phase, loss, epoch):
        if phase != 'train':
            self.writer.add_scalar(f'{phase}/loss_0', loss, epoch)


class EvalIterCb:
    def __init__(self):
        pass

    def on_iter(self, it, max_it, input, out, target, data_dict, ad, idx, phase, epoch):
        for fn in data_dict['target_filename']:
            name = fn.split('/')[-1]
            out_fn = os.path.join('./data/eval', name)
            print(out_fn)
            cv2.imwrite(out_fn, to_numpy(out)[...,::-1])
            cv2.imwrite(out_fn+'.target.jpg', to_numpy(target)[...,::-1])

    def on_epoch(self, phase, loss, epoch):
        pass


def save_splits(exper_dir, ds_train, ds_val):
    def write_list(path, data):
        with open(path, 'w') as f:
            for l in data:
                f.write(str(l))
                f.write('\n')

    for ds in ds_train.datasets:
        np.savetxt(os.path.join(exper_dir, 'train_view.txt'), np.vstack(ds.view_list))
        write_list(os.path.join(exper_dir, 'train_target.txt'), ds.target_list)

    for ds in ds_val.datasets:
        np.savetxt(os.path.join(exper_dir, 'val_view.txt'), np.vstack(ds.view_list))
        write_list(os.path.join(exper_dir, 'val_target.txt'), ds.target_list)



def ds_init_fn(worker_id):
    np.random.seed(int(time()))



def parse_image_size(string):
    error_msg = 'size must have format WxH'
    tokens = string.split('x')
    if len(tokens) != 2:
        raise argparse.ArgumentTypeError(error_msg)
    try:
        w = int(tokens[0])
        h = int(tokens[1])
        return w, h
    except ValueError:
        raise argparse.ArgumentTypeError(error_msg)


def parse_args(parser):
    args, _ = parser.parse_known_args()
    assert args.pipeline, 'set pipeline module'
    pipeline = get_module(args.pipeline)()
    pipeline.export_args(parser)

    # override defaults
    if args.config:
        with open(args.config) as f:
            config = yaml.full_load(f)

        parser.set_defaults(**config)

    return parser.parse_args(), parser.parse_args([])


def print_args(args, default_args):
    from huepy import bold, lightblue, orange, lightred, green, red

    args_v = vars(args)
    default_args_v = vars(default_args)
    
    print(bold(lightblue(' - ARGV: ')), '\n', ' '.join(sys.argv), '\n')
    # Get list of default params and changed ones    
    s_default = ''     
    s_changed = ''
    for arg in sorted(args_v.keys()):
        value = args_v[arg]
        if default_args_v[arg] == value:
            s_default += f"{lightblue(arg):>50}  :  {orange(value if value != '' else '<empty>')}\n"
        else:
            s_changed += f"{lightred(arg):>50}  :  {green(value)} (default {orange(default_args_v[arg] if default_args_v[arg] != '' else '<empty>')})\n"

    print(f'{bold(lightblue("Unchanged args")):>69}\n\n'
          f'{s_default[:-1]}\n\n'
          f'{bold(red("Changed args")):>68}\n\n'
          f'{s_changed[:-1]}\n')


def check_pipeline_attributes(pipeline, attributes):
    for attr in attributes:
        if not hasattr(pipeline, attr):
            raise AttributeError(f'pipeline missing attribute "{attr}"')


def try_save_dataset(save_dir, dataset, prefix):
    if hasattr(dataset[0], 'target_list'):
        with open(os.path.join(save_dir, f'{prefix}.txt'), 'w') as f:
            for ds in dataset:
                f.writelines('\n'.join(ds.target_list))
                f.write('\n')


def save_args(exper_dir, args, prefix):
    with open(os.path.join(exper_dir, f'{prefix}.yaml'), 'w') as f:
        yaml.dump(vars(args), f)


if __name__ == '__main__':
    parser = MyArgumentParser(conflict_handler='resolve')
    parser.add = parser.add_argument
    parser.add('--eval', action='store_bool', default=True)
    parser.add('--train', dest='eval', action='store_false')
    parser.add('--crop_size', type=parse_image_size, default='976x544')
    parser.add('--batch_size', type=int, default=8)
    parser.add('--batch_size_val', type=int, default=None, help='if not set, use batch_size')
    parser.add('--lr', type=float, default=1e-4)
    parser.add('--freeze_net', action='store_bool', default=False)
    parser.add('--eval_in_train', action='store_bool', default=False)
    parser.add('--eval_in_train_epoch', default=-1, type=int)
    parser.add('--eval_in_test',  action='store_bool', default=True)
    parser.add('--merge_loss', action='store_bool', default=True)
    parser.add('--net_ckpt', type=Path, default=None, help='neural network checkpoint')
    parser.add('--save_dir', type=Path, default='data/experiments')
    parser.add('--epochs', type=int, default=100)
    parser.add('--seed', type=int, default=2019)
    parser.add('--save_freq', type=int, default=1, help='save checkpoint each save_freq epoch')
    parser.add('--log_freq', type=int, default=5, help='print log each log_freq iter')
    parser.add('--log_freq_images', type=int, default=100)
    parser.add('--comment', type=str, default='', help='comment to experiment')
    parser.add('--paths_file', type=str)
    parser.add('--dataset_names', type=str, nargs='+')
    parser.add('--exclude_datasets', type=str, nargs='+')
    parser.add('--config', type=Path)
    parser.add('--use_mask', action='store_bool')
    parser.add('--pipeline', type=str, help='path to pipeline module')
    parser.add('--inference', action='store_bool', default=False)
    parser.add('--ignore_changed_args', type=str, nargs='+', default=['ignore_changed_args', 'save_dir', 'dataloader_workers', 'epochs', 'max_ds', 'batch_size_val'])
    parser.add('--multigpu', action='store_bool', default=False)
    parser.add('--dataloader_workers', type=int, default=0)
    parser.add('--max_ds', type=int, default=4, help='maximum datasets in DataLoader at the same time')
    parser.add('--reg_weight', type=float, default=0.)
    parser.add('--input_format', type=str)
    parser.add('--num_mipmap', type=int, default=5)
    parser.add('--net_size', type=int, default=4)
    parser.add('--input_channels', type=int, nargs='+')
    parser.add('--conv_block', type=str, default='gated')
    parser.add('--supersampling', type=int, default=1)
    parser.add('--use_mesh', action='store_bool', default=False)
    parser.add('--simple_name', action='store_bool', default=False)

    args, default_args = parse_args(parser)

    setup_environment(args.seed)

    if args.eval:
        iter_cb = EvalIterCb()
    else:
        if args.simple_name:
            args.ignore_changed_args += ['config', 'pipeline']
        exper_name = get_experiment_name(args, default_args, args.ignore_changed_args)
        exper_dir = make_experiment_dir(args.save_dir, postfix=exper_name)

        writer = SummaryWriter(log_dir=exper_dir, flush_secs=10)
        iter_cb = TrainIterCb(args, writer)

        setup_logging(exper_dir)

        print(f'experiment dir: {exper_dir}')

    print_args(args, default_args)

    args = eval_args(args)

    pipeline = get_module(args.pipeline)()
    pipeline.create(args)

    required_attributes = ['model', 'ds_train', 'ds_val', 'optimizer1', 'optimizer2', 'criterion']
    check_pipeline_attributes(pipeline, required_attributes)

    lr_scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer1, patience=6, factor=0.5, verbose=True)
    lr_scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(pipeline.optimizer2, patience=6, factor=0.5, verbose=True)

    if args.net_ckpt:
        print(f'LOAD NET CHECKPOINT {args.net_ckpt}')
        load_model_checkpoint(args.net_ckpt, pipeline.get_net())

    if args.dcgan_ckpt:
        print(f'LOAD DCGAN CHECKPOINT {args.dcgan_ckpt}')
        load_model_checkpoint(args.dcgan_ckpt, pipeline.get_dcgan())
    
    if hasattr(pipeline.model, 'ray_block') and pipeline.model.ray_block is not None:
        if hasattr(args, 'ray_block_ckpt') and args.ray_block_ckpt:
            print(f'LOAD RAY BLOCK CHECKPOINT {args.ray_block_ckpt}')
            load_model_checkpoint(args.ray_block_ckpt, pipeline.model.ray_block)

    if args.freeze_net:
        print('FREEZE NET')
        freeze(pipeline.get_net(), True)

    if args.eval:
        val_loss, val_psnr1, val_ssim1 = run_eval(0, pipeline, args, iter_cb)
        print('VAL LOSS', val_loss)
        print('VAL PSNR1', val_psnr1)
        print('VAL SSIM1', val_ssim1)

    else:
        try_save_dataset(exper_dir, pipeline.ds_train, 'train')
        try_save_dataset(exper_dir, pipeline.ds_val, 'val')

        save_args(exper_dir, args, 'args')
        save_args(exper_dir, default_args, 'default_args')

        for epoch in range(args.continue_epoch, args.epochs):
            print('### EPOCH', epoch)

            print('> TRAIN')

            train_loss, _, _ = run_train(epoch, pipeline, args, iter_cb)
            
            print('TRAIN LOSS', train_loss)

            print('> EVAL')

            val_loss, val_psnr1, val_ssim1 = run_eval(epoch, pipeline, args, iter_cb)

            if val_loss is not None:
                lr_scheduler1.step(val_loss)
                lr_scheduler2.step(val_loss)

            print('VAL LOSS', val_loss)
            print('VAL PSNR1', val_psnr1)
            print('VAL SSIM1', val_ssim1)

            if (epoch + 1) % args.save_freq == 0:
                save_pipeline(pipeline, os.path.join(exper_dir, 'checkpoints'), epoch, 0, args)
