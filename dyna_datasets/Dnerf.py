import warnings

import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions
from .color_utils import read_image
import kornia
from .base import BaseDataset
from  .hexplane_dataloader import get_test_dnerf_dataset,get_train_dataset_dnerf

class DNeRFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        regenerate=kwargs.get('regenerate',False)
        self.STATIC_ONLY=kwargs.get('static_only',False)
        self.use_importance_sampling= kwargs.get('use_importance_sampling',True)
        self.sample_stages=2
        self.ray_sampling_strategy=None # to be set by train_dynamic.py
        self.batch_size=None # to be set by ...
        print(f'N3DV_dataset_with_hexplane_dataloader, split={split}')
        cfg={
            'root_dir':root_dir,
            'downsample':1/downsample, #hexplane different from ngp_pl
            'time_scale':1,
        }

        file_=os.path.join(root_dir,f'useful_data_{split}.pt')

        if os.path.exists(file_) and not regenerate:
            useful_data=torch.load(file_)
        else:

            if split=='train':
                useful_data=get_train_dataset_dnerf(cfg)
            else:
                useful_data=get_test_dnerf_dataset(cfg)

            torch.save(useful_data,file_)

        self.importance=None

        self.setup_dataset(useful_data)



    def setup_dataset(self, useful_data):
        self.K = useful_data['K']
        self.poses = useful_data['poses']

        # visualize_poses(self.poses)

        self.times = useful_data['times']
        self.rays_rgbs = useful_data['rgb']
        self.rays = useful_data['rays']

        self.img_wh = useful_data['img_wh']
        # if self.split=='train' and self.use_importance_sampling:
        #    self.importance=useful_data['importance']#.numpy().astype(np.float64) # convert to double precision

        w, h = self.img_wh  # using ngp ray direction characteristics
        assert w >= h
        '''
        The ngp_pl repository have different means of calculating ray directions from nerf_pl.
         which leads to incorrect modeling and get ill-posed results.


        a naive idea is to use the nerf_pl get_ray_directions(H,W,F) function

        however, we should not do that, 

        because this ngp_pl repository uses poses to mark invisible cells... 

        a possible solution is to change "n3dv" dataset pose format into ngp_pl pose format

        '''
        # self.directions=useful_data['directions']

        self.directions = get_ray_directions(h, w, self.K)

        self.check_dimensions()
        self.current_epoch = 0
    def check_dimensions(self):

        self.times=self.times[:,0,:]
        self.N_time = len(self.times)

        self.poses=self.poses[:,:3,:]
        assert self.poses.shape==(self.N_time,3,4) # for D-nerf it is N_time  x 3 x 4

        self.W,self.H=self.img_wh[0],self.img_wh[1]
        assert self.W>=self.H

        self.rays=self.rays.view(self.N_time,self.W*self.H,6)
        assert self.rays.shape==(self.N_time,self.W*self.H,6)

        if self.split=='train':
            self.N_time=self.times.shape[0]
            self.rays_rgbs=self.rays_rgbs.view(self.N_time,self.W*self.H,3)


        else:
            self.N_time=len(self.times)
            self.rays_rgbs=self.rays_rgbs.view( self.N_time , self.W * self.H, 3)

        if self.STATIC_ONLY:

            self.ray_sampling_strategy='all_images'
            warnings.warn('static only ! setting ray sampling strategy to all images!')
            self.rays_rgbs=self.rays_rgbs.view(self.N_cam,self.N_time, self.W * self.H, 3)
            self.rays_rgbs=self.rays_rgbs[:,200:201,:,:]
            if self.split=='train':
                self.times = self.times[:, 200:201, :]
            else:
                self.times=self.times[200:201]
            self.N_time=1


    def __getitem__(self, idx):
        sample={}
        sample['t_trunk_size'] = self.batch_size

        if self.split.startswith('train'):
            '''
            self.rays_rgbs.shape == (self.N_cam,self.N_time* self.W * self.H, 3)
            self.importance.shape == (self.N_cam,self.N_time, self.W * self.H, 1)
            self.times: [N_time] tensor
            '''
            if self.ray_sampling_strategy =='hirachy':

                raise NotImplementedError


            elif self.ray_sampling_strategy == 'importance_time_batch':

                raise NotImplementedError

            else:



                # training pose is retrieved in train.py
                if self.ray_sampling_strategy == 'all_time': # randomly select across time
                    '''
                    the rendering module is focusd on "ray" instead of "3d point"
                    we only need to keep that every ray matches its time stamp. 
                    '''

                    time_indices = np.random.choice(self.N_time, self.batch_size,p=None,replace=True)
                    times =self.times[time_indices] # actually, for each camera it's identical
                elif self.ray_sampling_strategy == 'batch_time':
                    raise NotImplementedError

                else:
                    raise NotImplementedError
                    assert self.ray_sampling_strategy == 'same_time' # randomly select ONE time stamp
                    cam_idxs = np.random.choice(self.N_cam, self.batch_size,p=None,replace=True)

                    time_indices = np.random.choice(self.N_time,1)
                    times =self.times[cam_idxs,time_indices] # actually, for each camera it's identical

                    #time_indices = time_indices*np.ones(self.batch_size).astype(np.int)

                if self.STATIC_ONLY:
                    '''
                    only have one time stamp. the index is zero.
                    Does not mean the time value is zero!
                    '''
                    time_indices = np.zeros_like(time_indices) #should be zero-indices!
                    times = self.times[cam_idxs,time_indices] # actually, for each camera it's identical




                # importance sampling seems to comsume huge amount of memory...
                # consider implement it in a memory-efficient way
                #sampling_probs=self.importance.view(self.N_cam,self.N_time,self.H*self.W,1)
                #sampling_probs=sampling_probs[:,time_indices]
                #sampling_probs=sampling_probs[:]

                #sampling_probs=sampling_probs.numpy().astype(np.float64)

                #ray_intices=np.zeros(self.batch_size,dtype=int)
                #for i in range(self.batch_size):
                #    ray_intices[i]=np.random.choice(self.W*self.H,self.batch_size,p=None,replace=True)

                ray_indices = np.random.choice(self.W*self.H,self.batch_size,p=None,replace=True)

                rgbs = self.rays_rgbs.view( self.N_time, self.H * self.W, 3)[ time_indices, ray_indices]


                #print(f'times {times}')

                # randomly select pixels
                '''
                pixel indices is to get directions.
                '''
                pix_idxs = ray_indices


                #breakpoint()

                #print(f'im {cam_idxs.shape},pix{pix_idxs.shape},times{times.shape},rgb{rgbs.shape}')
                tmp = {'img_idxs': time_indices, 'pix_idxs': pix_idxs,
                          'times':times,
                          'rgb': rgbs[:, :3]}
                sample.update(tmp)
                #print(sample)

        else:


            # test time
            '''
            self.rays_rgbs.shape == (self.N_cam, self.N_time , self.W * self.H, 3)
            '''


            sample = {'pose': self.poses[0], 'img_idxs': idx,
                      'times':torch.Tensor([self.times[idx]])
                      }
            if len(self.rays_rgbs)>0: # if ground truth available
                rgbs = self.rays_rgbs.view(-1,self.H*self.W, 3)[idx,:,:] # stack the different cams and different times
                #sample['times']= torch.ones_like(rgbs)*self.times[idx]

                #print(f'rgbs{self.rays_rgbs},{self.rays_rgbs.shape}')
                sample['rgb'] = rgbs

        return sample

