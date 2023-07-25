import datetime

import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange
'''
set visible devices before initializing tcnn module.
to let run on 3090 GPU.
'''

ROOT_dir='/home/ubuntu/datasets/zhenhuanliu/ngp_pl/data/n3dv/cut_roasted_beef'

import socket
if socket.gethostname().startswith('zhenhuanliu'):
    # 227 workstation
    print(f'227 workstation')
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
else:
    assert 'ubuntu-Z690-GAMING-X-DDR4' in socket.gethostname()
    print(f'428 server')

    # professor's server
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# data
from torch.utils.data import DataLoader
from dyna_datasets import dataset_dict
from dyna_datasets.ray_utils import axisangle_to_R, get_rays,get_rays_hexplane_method

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks_dynamic import NGP_4D ,NGP_time
from models.networks import NGP
from models.networks_dynamic_plus import NGP_time_code_single as NGP_time_code
from models.networks_dynamic_simple import NGP_time_code_simple
from models.networks_dct import DCT_NGP

#from models.rendering import render, MAX_SAMPLES
from models.rendering_time import render,MAX_SAMPLES
# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss,loss_sum,dict_sum
from torch.optim.adam import Adam
# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure,MultiScaleStructuralSimilarityIndexMeasure
)
from pytorch_msssim import ms_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
#from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
#from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from utils import slim_ckpt, load_ckpt
from dyna_datasets.ray_utils import visualize_poses
#import warnings; warnings.filterwarnings("ignore")
from models.debug_utils import nan_dict_check,nan_check
import pynvml
visualize_poses_flag=False
import warnings
def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img

import re
def dir_to_indx(dir_str):
    pattern=re.compile(r'\d+')
    indx=pattern.search(dir_str)
    indx_start_in_str=indx.start()
    assert isinstance(indx_start_in_str,int)

    prefix=dir_str[:indx_start_in_str]

    indx_int=int(indx.group())

    return indx_int,prefix


# TODO: consider the following:
#    1. how to implement the background field. to compensate for the outdoor scenery.
#    2. using the importance sampling strategy, refer to hexplane code..
#    3. consider use the 2-stage training strategy...but may be very hard
#    4. consider remove the static branch and add flow consistency regularization...
# TODO: tune the dct-nerf.
#  1. how to deal with the cases that some of the motions are sometimes out of the grid bounds?
#  2. occupancy grids updating strategy...
#  3. vertex magnitude ... should be the same with different relationship? upper and down are the same?
#
#
#
#

class DNeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 4096 # almost 4096/300= 14 epochs
        self.update_interval = int(hparams.update_interval) # 8 or 16 seeming not to vary too much..
        self.distortion_loss_step = 300* 60 #if hparams.ray_sampling_strategy=='hirachy' else 300

        self.loss = NeRFLoss(lambda_opacity=self.hparams.opacity_loss_w,
                             lambda_entropy=self.hparams.entropy_loss_w,
                             sigma_entropy=self.hparams.sigma_entropy_loss_w,
                             lambda_opac_dyna=self.hparams.opacity_loss_dynamic_w,
            lambda_distortion=self.hparams.distortion_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        self.val_ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips_alex = LearnedPerceptualImagePatchSimilarity('alex')
            self.val_lpips_vgg = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips_alex.net.parameters():
                p.requires_grad = False
            for p in self.val_lpips_vgg.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        if self.hparams.model_type==0:
            self.model = NGP_time(scale=self.hparams.scale, rgb_act=rgb_act)
            from models.rendering import render as render_func
            self.render_function=render_func
        elif self.hparams.model_type==-1:
            #self.model = NGP_time_code_simple(scale=self.hparams.scale, rgb_act=rgb_act)
            self.model = NGP_4D(scale=self.hparams.scale, rgb_act=rgb_act)
            from models.rendering import render as render_func

            self.render_function=render_func
        else:
            from models.rendering_time import render as render_func
            self.render_function=render_func

            self.model = NGP_time_code(scale=self.hparams.scale, rgb_act=rgb_act)

        if not isinstance(self.model,(NGP_time_code,NGP_time_code_simple,DCT_NGP)):
            G = self.model.grid_size
            self.model.register_buffer('density_grid',
                torch.zeros(self.model.cascades, G**3))
            self.model.register_buffer('grid_coords',
                create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))


        self.process_cache={}
        self.process_cache['validation_epoch']=[]
        self.use_trunk_to_avoid_OOM=False
        self.val_version_dir=None
    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'regenerate':bool(self.hparams.regenerate),
                  'static_only':bool(self.hparams.static_only),
                  'downsample': self.hparams.downsample}
        self.train_dataset = dataset(split=self.hparams.split, **kwargs)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.sample_stages=2
        self.train_dataset.importance_sampling_size = self.train_dataset.batch_size * 2 // 4

        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy
        if self.hparams.ray_sampling_strategy=='importance_time_batch':
            self.train_dataset.generate_importance_sampling_indices(self.hparams.cache_importance_epochs)
        #self.train_dataset.set_t_resolution(1)
        pynvml.nvmlInit()
        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(1)

        self.test_dataset = dataset(split='test', **kwargs)
        torch.cuda.empty_cache()
    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))
        if visualize_poses_flag:
            visualize_poses(self.poses.to('cpu').numpy())

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        net_params = []
        for n, p in self.named_parameters():
            if 'lpips' in n:continue
            if n not in ['dR', 'dT']:
                #print(n)

                if 'net' in n:
                    wd=1e-7
                else:
                    wd=5e-8
                net_params += [
                    {
                        'params':p,
                        'weight_decay':wd
                    }
                ]
        #print(net_params)
        #exit(0)
        opts = []
        #https://github.com/NVlabs/tiny-cuda-nn/issues/219
        # use amsgrad for better numerical stability
        # apex afusedadam is better than pytorch adam, fused adam, amsgrad
        # but each implementation causes nan in some times.
        '''
        pytorch fused adam is significaltly (20%+) slower than apex fused adam.
        and with slightly worse reconstruction quality.
        consider double check later.
        do not tune the eps.
        to choose optimizer, just by commenting and dis-commenting the codes below. 
        to fix the numerical issue, reference: https://github.com/Lightning-AI/lightning/issues/15501
        may need to implement mannual backward loop.
        
        
        float32 will not run into numeric issue.
        float16 will be likely to run into numeric issue.
        
        add weight decay to avoid the parameter of nn become too large ( inf )
        '''

        #self.net_opt = Adam(net_params, self.hparams.lr,eps=1e-15,fused=True,#amsgrad=True,
        #                    weight_decay=5e-9 # think that weight decay = 1e-6 is definitely ok to avoid numerical issue, but with low reconstruction quality.
        self.net_opt = FusedAdam(net_params, self.hparams.lr,eps=1e-15,
                                 weight_decay=5e-8
                                 ) # ngp_pl repository set eps  1e-15, but may result in numerical error

        opts += [self.net_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        net_sch = CosineAnnealingLR(self.net_opt,
                                    T_max=self.hparams.num_epochs*4//10,
                                    eta_min=self.hparams.lr/30,
                                    last_epoch=-1
                                    )

        return opts, [net_sch]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          num_workers=16,
                          persistent_workers=True,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
                                        self.poses,
                                        self.train_dataset.img_wh)
    def on_train_epoch_end(self):

        self.model.fine_tune=False#(self.current_epoch> 120)
        if self.current_epoch==60:
            self.train_dataset.importance_sampling_size = self.train_dataset.batch_size * 2 // 4
            self.train_dataset.sample_stages=2
        elif self.current_epoch==120:
            self.train_dataset.importance_sampling_size = self.train_dataset.batch_size * 3 // 4
            self.train_dataset.sample_stages=2
        elif self.current_epoch==180:
            self.train_dataset.importance_sampling_size = self.train_dataset.batch_size * 7 // 8
            self.train_dataset.sample_stages=2
        elif self.current_epoch==300:
            self.train_dataset.importance_sampling_size = self.train_dataset.batch_size * 3 // 4
            self.train_dataset.sample_stages=3

        if self.hparams.ray_sampling_strategy != 'importance_time_batch':
            return
        self.train_dataset.current_epoch=self.current_epoch+1
        if (self.current_epoch+1)%self.train_dataset.importance_sampling_cache_epoches==0:
            if (self.current_epoch + 1)<self.hparams.num_epochs:
                # not the last epoch end
                self.train_dataset.generate_importance_sampling_indices(epochs=self.hparams.cache_importance_epochs)

    def training_step(self, batch, batch_nb, *args):
        if self.global_step%self.update_interval == 0:
            with torch.no_grad():
                self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
                                           warmup=self.global_step<self.warmup_steps,
                                           erode=bool(self.hparams.erode),
                      #                     erode=self.hparams.dataset_name=='colmap'
                                           )

        trunk_numbers= batch['times'].shape[0]//batch['t_trunk_size']

        '''
        compute loss for each trunk. sum up, and divide by the trunk numbers
        to avoid concat the results of trunks together (may be a lot of memory allocation, time consuming)
        just pass the "density grids/ bits index", and slice by the start to end in the rendering function.
        
        https://discuss.pytorch.org/t/does-indexing-a-tensor-return-a-copy-of-it/164905
        
        in the rendering function, just use some of the rows
        so that there will not be need to copy or re-create a variable
        '''
        summed_loss_trunk=None

        psnr_sum=0

        named_results={
        }

        for i in range(trunk_numbers):
            start_=i*batch['t_trunk_size']
            batch['start']=start_
            end_=(i+1)*batch['t_trunk_size']
            batch['end']=end_
            assert start_<end_
            batch['t_grid_indx']=i

            results_trunk = self(batch, split='train')


            summed_loss_trunk = loss_sum(loss_A=summed_loss_trunk,
                                   loss_B=self.loss(results_trunk, batch,use_dst_loss=self.global_step>=self.distortion_loss_step)
                                       )

            nan_dict_check(summed_loss_trunk)

            named_results=dict_sum(named_results,results_trunk,keys=['vr_samples','rm_samples'])
            with torch.no_grad():
                psnr_sum+=self.train_psnr(results_trunk['rgb'], batch['rgb'][start_:end_])

        if self.hparams.use_exposure:
            raise NotImplementedError
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            summed_loss_trunk['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in summed_loss_trunk.values()) / trunk_numbers

        try:
            nan_check(loss)
        except:
            exit(10)
        '''
        calculate every trunk psnr and average them.
        '''
        self.log('lr', self.net_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        self.log('train/entropy', summed_loss_trunk['entropy'].mean()/self.loss.lambda_entropy,True)
        self.log('train/sigma_entropy', summed_loss_trunk['sigma_entropy'].mean()/self.loss.lambda_sigma_entropy,True)
        self.log('train/opacity', summed_loss_trunk['opacity'].mean()/self.loss.lambda_opacity,True)
        self.log('train/opacity_dynamic', summed_loss_trunk['opacity_dynamic'].mean()/self.loss.lambda_opac_dyna,True)

        if 'distortion' in summed_loss_trunk:
            self.log('train/distortion', summed_loss_trunk['distortion'].mean() / self.loss.lambda_distortion, True)
        self.log('train/erode', self.hparams.erode)
        self.log('train/opacity_loss_w', self.hparams.opacity_loss_w)
        self.log('train/entropy_loss_w', self.hparams.entropy_loss_w)
        self.log('train/distortion_loss_w', self.hparams.distortion_loss_w)
        self.log('train/static_weight', named_results['static_weight_average']/len(named_results),True)

        info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)

        self.log('train/gpu_memory_usage_MB',info.used / 1024 / 1024 , True)

        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', named_results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', named_results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', psnr_sum/trunk_numbers,True)

        return loss

    def forward(self, batch, split):

        if split=='train':
            return self.forward_train(batch,split)
        else:
            assert split=='test'
            return self.forward_inference(batch,split)



    def forward_train(self,batch, split):

        start_=batch['start']
        end_=batch['end']

        #for i in batch.keys():
        #    print(i,batch[i])
        poses = self.poses[batch['img_idxs'][start_:end_]]
        directions = self.directions[batch['pix_idxs'][start_:end_]]
        times=batch['times'][start_:end_]
        if self.hparams.optimize_ext:
            raise NotImplementedError
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]
        #print(f'poses {self.poses},{self.poses.shape}'
        #      f'directions,{self.directions},{self.directions.shape}')
        rays_o, rays_d = get_rays(directions, poses)
        rays_o=rays_o.contiguous()
        rays_d=rays_d.contiguous()
        #print(f'rays_o{rays_o},{rays_o.shape}'
        #      f'rays_d{rays_d},{rays_d.shape}'
        #      f'')
        #rays_all=batch['rays']
        kwargs = {'test_time': split!='train',
                  'times':times,
                  'random_bg': self.hparams.random_bg}
        if self.hparams.dataset_name=='dnerf':
            kwargs['exp_step_factor'] = 0
        elif self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        if self.hparams.ray_sampling_strategy in ['batch_time','importance_time_batch','hirachy','all_times' ]:
            kwargs['t_grid_indx']= batch['t_grid_indx']

        #assert rays_o.shape[0]==rays_d.shape[0]==self.train_dataset.batch_size

        return self.render_function(self.model, rays_o, rays_d, **kwargs)


    def forward_inference(self,batch, split):

        poses = batch['pose']
        directions = self.directions
        times=batch['times']

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'times': times.to(self.device),
                  'random_bg': self.hparams.random_bg}
        '''
        increase trunk size to get better inference performance!
        '''

        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']
        try:
            assert not self.use_trunk_to_avoid_OOM
            result_ = render(self.model, rays_o[:], rays_d[:], **kwargs)
        except:
            warnings.warn('use trunks to avoid OOM! performance may be downgraded!')
            kwargs['trunks'] = 32768 * 4096
            '''
            increase trunk size to get better inference performance!
            '''
            result_ = render(self.model, rays_o[:], rays_d[:], **kwargs)
            self.use_trunk_to_avoid_OOM = True
        return result_

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            val_dir_root = f'results/dynamic/{self.hparams.dataset_name}/{self.hparams.exp_name}'
            os.makedirs(val_dir_root, exist_ok=True)
            sub_paths = os.listdir(val_dir_root)
            indices = []
            if len(sub_paths) == 0:
                new_indx = 0
            else:
                for path_ in sub_paths:
                    if os.path.isdir(os.path.join(val_dir_root,path_)):
                        indx, prefix = dir_to_indx(path_)

                        indices += [indx]
                try:
                    max_indx = max(indices)
                    new_indx = max_indx + 1

                except:

                    new_indx = 0
            prefix = 'version_'

            if self.val_version_dir is None:
                self.val_version_dir = prefix + f'{new_indx}'



            self.val_dir = os.path.join(val_dir_root, self.val_version_dir, f'epoch_{self.current_epoch}')
            os.makedirs(self.val_dir, exist_ok=False)
            print(f'{ datetime.datetime.now()}, validation result of epoch{self.current_epoch} will be saved at {self.val_dir}')

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']

        #print(f'rgb_gt {rgb_gt.shape}')


        trunk= 16384



        results = self(batch, split='test')
        #print(f'validation results{results}')

        #print(f"results['rgb'] {results['rgb'].shape}")

        logs = {}
        # compute each metric per image

        assert results['rgb'].shape==rgb_gt.shape
        self.val_psnr(results['rgb'], rgb_gt)
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        logs['batch_size']=self.hparams.batch_size

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)

        logs['ssim'] = self.val_ssim.compute()
        ms_ssim_2=ms_ssim(rgb_pred,rgb_gt,data_range=1)
        logs['ms_ssim'] = ms_ssim_2

        logs['d_ssim'] = (1-logs['ms_ssim'])/2.0
        info =pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
        logs['gpu_mem'] = info.used / 1024 /1024

        self.val_ssim.reset()


        if self.hparams.eval_lpips:
            self.val_lpips_alex(torch.clip(rgb_pred * 2 - 1, -1, 1),
                                torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips_alex'] = self.val_lpips_alex.compute()
            self.val_lpips_alex.reset()

            self.val_lpips_vgg(torch.clip(rgb_pred * 2 - 1, -1, 1),
                                torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips_vgg'] = self.val_lpips_vgg.compute()
            self.val_lpips_vgg.reset()


        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'rgb_{idx:04d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'depth_{idx:04d}.png'), depth)

        self.process_cache['validation_epoch'].append(logs)
        return logs
    def on_validation_epoch_end(self):
        outputs=self.process_cache['validation_epoch']
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = psnrs.mean()
        self.log('test/psnr', mean_psnr, True)

        ssims = torch.stack([x['ssim'] for x in outputs])
        ms_ssims= torch.stack([x['ms_ssim'] for x in outputs])
        gpu_mem= max(([x['gpu_mem'] for x in outputs]))

        mean_ssim = ssims.mean()
        self.log('test/ssim', mean_ssim)
        self.log('test/ms_ssim', ms_ssims.mean())
        self.log('test/d_ssim', (1- ms_ssims.mean())/2.0)
        self.log('test/gpu_mem', gpu_mem)

        if self.hparams.eval_lpips:
            lpipss_vgg = torch.stack([x['lpips_vgg'] for x in outputs])
            mean_lpips_vgg = lpipss_vgg.mean()
            self.log('test/lpips_vgg', mean_lpips_vgg)
            lpipss_alex = torch.stack([x['lpips_alex'] for x in outputs])
            mean_lpips_alex = lpipss_alex.mean()
            self.log('test/lpips_alex', mean_lpips_alex)



        imgs = sorted(glob.glob(os.path.join(self.val_dir, 'rgb_*.png')))


        imageio.mimsave(os.path.join(self.val_dir, 'rgb.mp4'),
                        [imageio.v2.imread(img) for img in imgs],
                        fps=30, macro_block_size=1)
        imgs_depth = sorted(glob.glob(os.path.join(self.val_dir, 'depth_*.png')))

        imageio.mimsave(os.path.join(self.val_dir, 'depth.mp4'),
                        [imageio.v2.imread(img) for img in imgs_depth],
                        fps=30, macro_block_size=1)

        #self.process_cache.pop()
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
def print_time_elapse(t0,t1,prefix=''):
    dt = (t1 - t0).seconds
    dmins = np.floor(dt / 60)
    print(f'{prefix } = {dt} seconds,i.e.{dmins} min {dt - dmins * 60} s')


if __name__ == '__main__':

    device_=torch.cuda.get_device_name(0)
    #assert device_.endswith('3090')
    #torch.cuda.memory_summary(device=None, abbreviated=False)
    hparams = get_opts()
    t_start=datetime.datetime.now()
    #torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=True #

    print(f'{datetime.datetime.now()}')
    print(f'configs={hparams}')
    if hparams.val_only and (not hparams.ckpt_path):
        raise ValueError('You need to provide a @ckpt_path for validation!')
    system = DNeRFSystem(hparams)
    compiled_system=system

    #compiled_system=torch.compile(system,backend="eager") # pytorch compile not compatible with tcnn
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<16>"
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=max(1,hparams.num_epochs//5),
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/dynamic/{hparams.dataset_name}",
                               name=hparams.exp_name,
                               default_hp_metric=False)



    trainer = Trainer(max_epochs=hparams.num_epochs,
                      check_val_every_n_epoch=max(1,hparams.num_epochs//10),#min(5,max(1,hparams.num_epochs//5)),
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      benchmark=True,
         #             gradient_clip_val=10000,gradient_clip_algorithm='value',
                      precision='16-mixed')
    t0=datetime.datetime.now()
    print(f'{datetime.datetime.now()},start traning.')
    #torch._dynamo.config.verbose = True
    #torch._dynamo.config.suppress_errors = True
    torch.cuda.empty_cache()
    trainer.fit(compiled_system, ckpt_path=hparams.ckpt_path)

    t1=datetime.datetime.now()


    '''
    
    speed: 
    
    hexplanes: NVIDIA V100 2 hours (SOTA)  approx. RTX 3090 1.x hour
    K-Planes: NVIDIA A30  ~4 hours
    
    mine: NVIDIA RTX 3090 ~ 1 hours is almost OK.
    
    
    '''
    print_time_elapse(t0,t1,'training + evaluation')
    print_time_elapse(t_start,t1,'data preparing + training + evaluation')



    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       'Synthetic' in hparams.root_dir: # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.v2.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.v2.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
