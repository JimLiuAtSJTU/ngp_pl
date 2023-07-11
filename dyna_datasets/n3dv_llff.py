import datetime
import warnings

import torch
import json
import numpy as np
import os
from tqdm import tqdm

from .ray_utils import get_ray_directions,visualize_poses
from .color_utils import read_image

from .base import BaseDataset
from .hexplane_dataloader import get_test_dataset,get_train_dataset
from .importance_sampling.Sampling import GM_Resi,GM_function

STATIC_ONLY=False #debug only

val_indx_N3DV=0



'''
poses bounds.npy #https://github.com/Fyusion/LLFF

'''



class N3DV_dataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)
        regenerate=kwargs.get('regenerate',False)
        self.use_importance_sampling= kwargs.get('use_importance_sampling',True)

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
                useful_data=get_train_dataset(cfg)
            else:
                useful_data=get_test_dataset(cfg)

            torch.save(useful_data,file_)

        self.importance=None

        self.setup_dataset(useful_data)



    def setup_dataset(self, useful_data):
        self.K=useful_data['K']
        self.poses=useful_data['poses']

        #visualize_poses(self.poses)

        self.times=useful_data['times']
        self.rays_rgbs=useful_data['rgb']
        self.rays=useful_data['rays']

        self.img_wh=useful_data['img_wh']
        if self.split=='train' and self.use_importance_sampling:
            self.importance=useful_data['importance']#.numpy().astype(np.float64) # convert to double precision

        w,h=self.img_wh #using ngp ray direction characteristics
        assert w>h
        '''
        The ngp_pl repository have different means of calculating ray directions from nerf_pl.
         which leads to incorrect modeling and get ill-posed results.
         
        
        a naive idea is to use the nerf_pl get_ray_directions(H,W,F) function
        
        however, we should not do that, 
        
        because this ngp_pl repository uses poses to mark invisible cells... 
        
        a possible solution is to change "n3dv" dataset pose format into ngp_pl pose format
        
        '''
        #self.directions=useful_data['directions']

        self.directions = get_ray_directions(h, w, self.K)

        self.check_dimensions()
        self.current_epoch=0
    def check_dimensions(self):
        self.N_cam=self.poses.shape[0]
        assert self.poses.shape==(self.N_cam,3,4)

        self.W,self.H=self.img_wh[0],self.img_wh[1]
        assert self.W>self.H

        assert self.rays.shape==(self.N_cam,self.W*self.H,6)

        if self.split=='train':
            self.N_time=self.times.shape[1]

            assert self.rays_rgbs.shape == (self.N_cam,self.N_time* self.W * self.H, 3)



            try:
                assert self.importance.shape == (self.N_cam,self.N_time, self.W * self.H, 1) or self.importance is None
            except:
                self.importance=self.importance.view(self.N_cam,self.N_time, self.W * self.H, 1)


            if not isinstance(self.importance,torch.Tensor):
                self.importance = torch.from_numpy(self.importance)

            with torch.no_grad():
                '''
                use double precision to let the probabilities summing to 1.
                '''
                self.global_mean = torch.mean(self.rays_rgbs.to(0),dim=0)


                self.importance = self.importance.to(0).double()
                std_mean_=torch.std_mean(self.importance)
                max_, min_ =torch.max(self.importance),torch.min(self.importance)
                flag = False
                if flag:
                    stage_1_gamma = 0.001
                    stage_2_gamma = 0.02  # from hexplane code
                    stage_3_alpha = 0.1

                    prob_rays = self.rays_rgbs.view(self.N_cam, self.N_time, -1, 3)

                    self.prob_tmp = torch.zeros_like(prob_rays)
                    for i in range(self.N_cam):
                        self.prob_tmp[i] = GM_Resi(
                            prob_rays[i], self.global_mean[i], stage_1_gamma
                        )


                #self.importance = torch.pow(self.importance, 0.8)
                self.importance /= torch.sum(self.importance)
            self.global_mean=self.global_mean.to('cpu')

            self.importance=self.importance.to('cpu').numpy().astype(np.float64)


        else:
            self.N_time=len(self.times)
            assert self.rays_rgbs.shape == (self.N_cam, self.N_time , self.W * self.H, 3)

        if STATIC_ONLY:


            self.importance=None
            self.rays_rgbs=self.rays_rgbs.view(self.N_cam,self.N_time, self.W * self.H, 3)
            self.rays_rgbs=self.rays_rgbs[:,200:201,:,:]
            if self.split=='train':
                self.times = self.times[:, 200:201, :]
            else:
                self.times=self.times[200:201]
            self.N_time=1

    def set_t_resolution(self,reso):
        self.t_resolution=reso
    def stratify_sample(self):
        '''
        use time stratified sampling is better
        '''

        time_grid_indices0=np.arange(self.t_resolution)


        time_indices = np.zeros(self.batch_size)
        chunk_size = self.batch_size // self.t_resolution

        #assert self.N_time % self.t_resolution ==0
        assert self.batch_size % self.t_resolution ==0 # the function may be extended afterwards


        for i in time_grid_indices0:

            single_range_size=(int(np.ceil(self.N_time/self.t_resolution)))


            if i <self.t_resolution-1: #not the last
                time_range=np.arange(single_range_size*i,single_range_size+single_range_size*i)
            else:
                time_range=np.arange(single_range_size*i,self.N_time)


            time_indices[i*chunk_size:i*chunk_size+chunk_size]=np.random.choice(time_range, chunk_size,p=None,replace=True)


        return time_indices,chunk_size

    def stratify_importance_sample(self):
        '''
        use time stratified sampling is better
        '''
        # not clearly tested
        raise  NotImplementedError
        sample={}
        time_grid_indices0=np.arange(self.t_resolution)


        cam_idxs = np.zeros(self.batch_size)
        time_indices = np.zeros(self.batch_size)
        ray_indices = np.zeros(self.batch_size)

        chunk_size = self.batch_size // self.t_resolution

        #assert self.N_time % self.t_resolution ==0
        assert self.batch_size % self.t_resolution ==0 # the function may be extended afterwards

        for i in time_grid_indices0:

            single_range_size=(int(np.ceil(self.N_time/self.t_resolution)))


            if i <self.t_resolution-1: #not the last
                time_range=np.arange(single_range_size*i,single_range_size+single_range_size*i)
            else:
                time_range=np.arange(single_range_size*i,self.N_time)

            '''
            self.importance.shape == (self.N_cam,self.N_time, self.W * self.H, 1)
            '''

            time_grid_importance = self.importance[:,time_range[0]:time_range[-1],:,:]
            time_grid_importance=time_grid_importance.view(self.N_cam,time_range[-1]-time_range[0],self.W*self.H)
            time_grid_importance=time_grid_importance/np.sum(time_grid_importance)
            flatten_improtance = time_grid_importance.view(self.N_cam * (time_range[-1]-time_range[0]) * self.W * self.H)
            flatten_indices = np.random.choice(a=flatten_improtance.shape[0], size=chunk_size, replace=False,
                                               p=flatten_improtance)

            structured_rgb = self.rays_rgbs[self.N_cam,time_range[0]:time_range[-1], self.W * self.H, 3]\
                .view(self.N_cam, time_range[-1]-time_range[0], self.W * self.H, 3)
            structured_rgb_indices = np.unravel_index(indices=flatten_indices, shape=structured_rgb.shape[:3])
            cam_idxs_trunk, time_indices_trunk, ray_indices_trunk = \
                structured_rgb_indices[0], structured_rgb_indices[1], structured_rgb_indices[2]

            cam_idxs[i*chunk_size:i*chunk_size+chunk_size] = cam_idxs_trunk
            time_indices[i*chunk_size:i*chunk_size+chunk_size] = time_indices_trunk + time_range[0]
            ray_indices[i*chunk_size:i*chunk_size+chunk_size] = ray_indices_trunk

        times = self.times[cam_idxs, time_indices]
        rgbs = self.rays_rgbs.view(self.N_cam, self.N_time, self.H * self.W, 3)[cam_idxs, time_indices, ray_indices]

        tmp = {'img_idxs': cam_idxs, 'pix_idxs': ray_indices,
               'times': times,
               'rgb': rgbs[:, :3]}
        sample.update(tmp)

        return time_indices,chunk_size

    def hexplane_sample(self):

        self.sample_stages=2

        key_f_num=30
        stage_1_gamma= 0.001
        stage_2_gamma= 0.001 # 0.02 from hexplane code
        stage_3_alpha= 0.01
        # self.sample_stages
        if self.sample_stages==1:
            # Stage 1: randomly sample a single image from an arbitrary camera.
            # And sample a batch of rays from all the rays of the image based on the difference of global median and local values.
            # Stage 1 only samples key-frames, which is the frame every self.cfg.data.key_f_num frames.

            '''
                self.rays_rgbs.shape == (self.N_cam,self.N_time* self.W * self.H, 3)
                self.importance.shape == (self.N_cam,self.N_time, self.W * self.H, 1)
                self.times: [N_time] tensor
            '''

            cam_i = np.random.choice(self.rays_rgbs.shape[0])
            t_index_i = np.random.choice(
                self.N_time // key_f_num
            )
            rgb_train_importance = (
                self.rays_rgbs.view(self.N_cam,self.N_time,-1,3)[cam_i, t_index_i * key_f_num]
                .view(-1, 3)
            )

            frame_time = self.times[ cam_i, t_index_i * key_f_num]

            # Calcualte the probability of sampling each ray based on the difference of global median and local values.
            probability = GM_Resi(
                rgb_train_importance, self.global_mean[cam_i], stage_1_gamma
            )
            print(probability.shape)
            select_inds = torch.multinomial(
                probability, self.batch_size//2
            )





            rgb_train_importance = rgb_train_importance[select_inds]

            im_idx= torch.full((self.batch_size//2,),cam_i,dtype=torch.int32)
            times_= frame_time.expand(self.batch_size//2)
            tmp = {'img_idxs': im_idx, 'pix_idxs': select_inds,
                   'times': times_,
                   'rgb': rgb_train_importance[:, :3]}





        elif (
                self.sample_stages == 2
        ):
            # Stage 2: basically the same as stage 1, but samples all the frames instead of only key-frames.

            '''
                self.rays_rgbs.shape == (self.N_cam,self.N_time* self.W * self.H, 3)
                self.importance.shape == (self.N_cam,self.N_time, self.W * self.H, 1)
                self.times: [N_time] tensor
            '''

            cam_i = np.random.choice(self.rays_rgbs.shape[0])
            t_index_i = np.random.choice(
                self.N_time
            )
            rgb_train_importance = (
                self.rays_rgbs.view(self.N_cam,self.N_time,-1,3)[cam_i, t_index_i ]
                .view(-1, 3)
            )

            frame_time = self.times[ cam_i, t_index_i ]

            # Calcualte the probability of sampling each ray based on the difference of global median and local values.
            probability = GM_Resi(
                rgb_train_importance, self.global_mean[cam_i], stage_2_gamma
            )
            imp_sample_size =self.batch_size//3 * 2
            select_inds = torch.multinomial(
                probability, imp_sample_size
            )

            rgb_train_importance = rgb_train_importance[select_inds]

            cam_idxs_2 = np.random.choice(self.N_cam, self.batch_size - imp_sample_size, p=None, replace=True)
            time_indices_2 = np.random.choice(self.N_time, self.batch_size - imp_sample_size, p=None, replace=True)
            ray_indices_2 = np.random.choice(self.W * self.H, self.batch_size - imp_sample_size, p=None, replace=False)

            im_idx= torch.full((imp_sample_size,),cam_i,dtype=torch.int32)
            times_=torch.full((imp_sample_size,),frame_time.item(),dtype=torch.float)[:,None]


            cam_idxs = np.concatenate([im_idx, cam_idxs_2], axis=0)
            times_ = np.concatenate([ times_, self.times[cam_idxs_2, time_indices_2]], axis=0)
            ray_indices = np.concatenate([select_inds, ray_indices_2], axis=0)

            rgbs = self.rays_rgbs.view(self.N_cam, self.N_time, self.H * self.W, 3)[cam_idxs_2, time_indices_2, ray_indices_2]
            rgb_train_importance=torch.cat([rgb_train_importance,rgbs],dim=0)
            tmp = {'img_idxs': cam_idxs, 'pix_idxs': ray_indices,
                   'times': times_,
                   'rgb': rgb_train_importance[:, :3]}




        else:
            # Stage 3: randomly sample one frame and sample a batch of rays from the sampled frame.
            # TO sample a batch of rays from this frame, we calcualate the value changes of rays compared to nearby timesteps, and sample based on the value changes.
            cam_i = np.random.choice(self.rays_rgbs.shape[0])
            N_time = self.N_time
            # Sample two adjacent time steps within a range of 25 frames.
            t_index_i = np.random.choice(N_time)
            index_2 = np.random.choice(
                min(N_time, t_index_i + 25) - max(t_index_i - 25, 0)
            ) + max(t_index_i - 25, 0)
            rgb_train_importance = (
                self.rays_rgbs.view(self.N_cam,self.N_time,-1,3)[cam_i, t_index_i ]
            )
            rgb_ref = (
                self.rays_rgbs[cam_i, index_2].view(-1, 3)
            )
            rays_train = self.rays_rgbs[cam_i].view(-1, 6)
            frame_time = self.times[cam_i, t_index_i]
            # Calcualte the temporal difference between the two frames as sampling probability.
            probability = torch.clamp(
                1 / 3 * torch.norm(rgb_train_importance - rgb_ref, p=1, dim=-1),
                min=stage_3_alpha,
            )
            imp_sample_size =self.batch_size//3 * 2

            select_inds = torch.multinomial(
                probability, imp_sample_size
            ).to(rays_train.device)




            rgb_train_importance = rgb_train_importance[select_inds]

            cam_idxs_2 = np.random.choice(self.N_cam, self.batch_size -imp_sample_size, p=None, replace=True)
            time_indices_2 = np.random.choice(self.N_time, self.batch_size -imp_sample_size, p=None, replace=True)
            ray_indices_2 = np.random.choice(self.W * self.H, self.batch_size -imp_sample_size, p=None, replace=True)

            im_idx= torch.full((imp_sample_size,),cam_i,dtype=torch.int32)
            times_=torch.full((imp_sample_size,),frame_time.item(),dtype=torch.float)[:,None]

            cam_idxs = np.concatenate([im_idx, cam_idxs_2], axis=0)
            times_ = np.concatenate([ times_, self.times[cam_idxs_2, time_indices_2]], axis=0)
            ray_indices = np.concatenate([select_inds, ray_indices_2], axis=0)

            rgbs = self.rays_rgbs.view(self.N_cam, self.N_time, self.H * self.W, 3)[cam_idxs_2, time_indices_2, ray_indices_2]
            rgb_train_importance=torch.cat([rgb_train_importance,rgbs],dim=0)
            tmp = {'img_idxs': cam_idxs, 'pix_idxs': ray_indices,
                   'times': times_,
                   'rgb': rgb_train_importance[:, :3]}



        return tmp

    def generate_importance_sampling_indices(self,epochs=10):
        t0 = datetime.datetime.now()

        self.importance_sampling_cache_epoches=epochs
        flatten_improtance = self.importance.reshape(self.N_cam * self.N_time * self.W * self.H)
        print(flatten_improtance.shape)
        print(flatten_improtance.flags)
        print(np.sum(flatten_improtance))
        print(f'start sampling, size= {self.N_time*epochs}*{self.batch_size//2}')

        flatten_indices = np.random.choice(flatten_improtance.shape[0], size=self.batch_size//2 * self.N_time*epochs,
                                           p=flatten_improtance)
        t1 = datetime.datetime.now()

        print(f'end sampling, time epalse ={(t1 - t0).seconds}')

        self.cached_faltten_indices=(flatten_indices).reshape(epochs*self.N_time,self.batch_size//2)

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

                tmp=  self.hexplane_sample()
                #for k,v in tmp.items():
                #    print(k,v.shape)
                sample.update(tmp)


            elif self.ray_sampling_strategy == 'importance_time_batch':

                flatten_indices=self.cached_faltten_indices[idx + self.current_epoch*len(self),:]
                structured_rgb=self.rays_rgbs.view(self.N_cam,self.N_time, self.W * self.H, 3)
                structured_rgb_indices=np.unravel_index(indices=flatten_indices,shape=structured_rgb.shape[:3])
                cam_idxs, time_indices, ray_indices =\
                    structured_rgb_indices[0],structured_rgb_indices[1],structured_rgb_indices[2]

                cam_idxs_2 = np.random.choice(self.N_cam, self.batch_size//2, p=None, replace=True)
                time_indices_2 = np.random.choice(self.N_time, self.batch_size//2, p=None, replace=True)
                ray_indices_2 = np.random.choice(self.W*self.H,self.batch_size//2,p=None,replace=True)

                cam_idxs=np.concatenate([cam_idxs,cam_idxs_2],axis=0)
                time_indices=np.concatenate([time_indices,time_indices_2],axis=0)
                ray_indices=np.concatenate([ray_indices,ray_indices_2],axis=0)


                times=self.times[cam_idxs,time_indices]
                sample['t_trunk_size'] = self.batch_size
                rgbs = self.rays_rgbs.view(self.N_cam, self.N_time, self.H * self.W, 3)[cam_idxs, time_indices, ray_indices]

                tmp = {'img_idxs': cam_idxs, 'pix_idxs': ray_indices,
                          'times':times,
                          'rgb': rgbs[:, :3]}
                sample.update(tmp)

            else:



                # training pose is retrieved in train.py
                if self.ray_sampling_strategy == 'all_time': # randomly select across time
                    '''
                    the rendering module is focusd on "ray" instead of "3d point"
                    we only need to keep that every ray matches its time stamp. 
                    '''
                    cam_idxs = np.random.choice(self.N_cam, self.batch_size,p=None,replace=True)
                    time_indices = np.random.choice(self.N_time, self.batch_size,p=None,replace=True)
                    times =self.times[cam_idxs,time_indices] # actually, for each camera it's identical
                elif self.ray_sampling_strategy == 'batch_time':
                    # randomly select across time, but with a smaller batch size
                    # to use with  time - occupancy grids

                    cam_idxs = np.random.choice(self.N_cam, self.batch_size,p=None,replace=True)
                    time_indices,t_trunk_size=self.stratify_sample()
                    sample['t_trunk_size']=t_trunk_size
                    times =self.times[cam_idxs,time_indices] # actually, for each camera it's identical

                else:
                    assert self.ray_sampling_strategy == 'same_time' # randomly select ONE time stamp
                    cam_idxs = np.random.choice(self.N_cam, self.batch_size,p=None,replace=True)

                    time_indices = np.random.choice(self.N_time,1)
                    times =self.times[cam_idxs,time_indices] # actually, for each camera it's identical

                    #time_indices = time_indices*np.ones(self.batch_size).astype(np.int)

                if STATIC_ONLY:
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

                rgbs = self.rays_rgbs.view(self.N_cam, self.N_time, self.H * self.W, 3)[cam_idxs, time_indices, ray_indices]


                #print(f'times {times}')

                # randomly select pixels
                '''
                pixel indices is to get directions.
                '''
                pix_idxs = ray_indices


                #breakpoint()

                #print(f'im {cam_idxs.shape},pix{pix_idxs.shape},times{times.shape},rgb{rgbs.shape}')
                tmp = {'img_idxs': cam_idxs, 'pix_idxs': pix_idxs,
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

    #def __len__(self):
    #    if self.split.startswith('train'):
        #    return self.N_cam*self.N_time* self.W * self.H
        #return self.N_time*self.N_cam
    def __len__(self):
        if STATIC_ONLY:
            return self.poses.shape[0]

        return self.N_time
