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


val_indx_N3DV=0

class N3DV_dataset(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

        self.read_intrinsics()

        if kwargs.get('read_meta', True):
            self.read_meta(split)

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
            meta = json.load(f)

        assert isinstance(meta,dict)

        if 'w' in meta.keys():
            # for n3dv should be in meta
            w=int(meta['w']*self.downsample)
            h=int(meta['h']*self.downsample)
        else:
            w = h = int(800*self.downsample)

        if 'camera_angle_x' in meta.keys():


            fx = fy = 0.5*800/np.tan(0.5*meta['camera_angle_x'])*self.downsample
        else:
            assert 'fl_x' in meta.keys()
            assert 'fl_y' in meta.keys()
            fx,fy=meta['fl_x'], meta['fl_y']

        K = np.float32([[fx, 0, w/2],
                        [0, fy, h/2],
                        [0,  0,   1]])
        print(np.linalg.cond(K))
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)
    torch.autograd.set_detect_anomaly(True)
    def read_meta(self, split):
        rays = []
        poses = []
        times = []
        if split == 'trainval':
            with open(os.path.join(self.root_dir, "transforms_train.json"), 'r') as f:
                frames = json.load(f)["frames"]
            with open(os.path.join(self.root_dir, "transforms_val.json"), 'r') as f:
                frames+= json.load(f)["frames"]
        else:
            with open(os.path.join(self.root_dir, f"transforms_{split}.json"), 'r') as f:
                frames = json.load(f)["frames"]

        print(f'Loading {len(frames)} {split} images ...')
        self.rays = []
        self.poses = []
        self.times =[]


        if os.path.exists(os.path.join(self.root_dir,f'clean_data_{split}.pt')):
            file_=torch.load(os.path.join(self.root_dir,f'clean_data_{split}.pt'))
            self.rays,self.poses,self.times=file_[0],file_[1],file_[2]
            print(f'successfully loaded pre-generated data')

            return
        for frame in tqdm(frames):
            c2w = np.array(frame['transform_matrix'])[:3, :4]
            time_t= frame['time']
            #print(np.linalg.cond(c2w))
            # determine scale
            if 'Jrender_Dataset' in self.root_dir:
                c2w[:, :2] *= -1 # [left up front] to [right down front]
                folder = self.root_dir.split('/')
                scene = folder[-1] if folder[-1] != '' else folder[-2]
                if scene=='Easyship':
                    pose_radius_scale = 1.2
                elif scene=='Scar':
                    pose_radius_scale = 1.8
                elif scene=='Coffee':
                    pose_radius_scale = 2.5
                elif scene=='Car':
                    pose_radius_scale = 0.8
                else:
                    pose_radius_scale = 1.5
            else:
                c2w[:, 1:3] *= -1 # [right up back] to [right down front]
                pose_radius_scale = 1.5
            c2w[:, 3] /= np.linalg.norm(c2w[:, 3])/pose_radius_scale

            # add shift
            if 'Jrender_Dataset' in self.root_dir:
                if scene=='Coffee':
                    c2w[1, 3] -= 0.4465
                elif scene=='Car':
                    c2w[0, 3] -= 0.7
            poses += [c2w]
            times += [time_t]

            try:

                filename=frame['file_path']
                assert isinstance(filename,str)
                if not filename.endswith('jpg') or not filename.endswith('png'):
                    img_path = os.path.join(self.root_dir, f"{frame['file_path']}")
                else:
                    img_path = os.path.join(self.root_dir, f"{frame['file_path']}.png")
                img = read_image(img_path, self.img_wh)
                rays += [img]
            except:
                warnings.warn('something wrong while reading image')
        '''
        use temporary variable to avoid memory leak
        '''
        if len(rays)>0:
            self.rays = torch.FloatTensor(np.stack(rays)) # (N_images, hw, ?)
        self.poses = torch.FloatTensor(poses) # (N_images, 3, 4)
        self.times = torch.FloatTensor(times) # (N_images, 1)

        data_list=[self.rays,self.poses,self.times]
        torch.save(data_list,os.path.join(self.root_dir,f'clean_data_{split}.pt'))


class N3DV_dataset_2(BaseDataset):
    def __init__(self, root_dir, split='train', downsample=1.0, **kwargs):
        super().__init__(root_dir, split, downsample)

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

        if os.path.exists(file_):
            useful_data=torch.load(file_)
        else:

            if split=='train':
                useful_data=get_train_dataset(cfg)
            else:
                useful_data=get_test_dataset(cfg)

            torch.save(useful_data,file_)

        self.importance=None

        self.set_useful_data(useful_data)



    def set_useful_data(self,useful_data):
        self.K=useful_data['K']
        self.poses=useful_data['poses']

        #visualize_poses(self.poses)

        #self.directions=useful_data['directions']
        self.times=useful_data['times']
        self.rays_rgbs=useful_data['rgb']
        self.rays=useful_data['rays']

        self.img_wh=useful_data['img_wh']
        if self.split=='train' and self.use_importance_sampling:
            self.importance=useful_data['importance'].numpy().astype(np.float64) # convert to double precision

        h,w=self.img_wh #using ngp ray direction characteristics
        '''
        rays is NOT used.
        instead, use h,w,K to get directions for ngp_pl repository.
        '''
        self.directions = get_ray_directions(h, w, self.K)

        self.check_dimensions()
    def check_dimensions(self):
        self.N_cam=self.poses.shape[0]
        assert self.poses.shape==(self.N_cam,3,4)

        self.H,self.W=self.img_wh[0],self.img_wh[1]

        assert self.rays.shape==(self.N_cam,self.W*self.H,6)

        if self.split=='train':
            self.N_time=self.times.shape[1]

            assert self.rays_rgbs.shape == (self.N_cam,self.N_time* self.W * self.H, 3)
            assert self.importance.shape == (self.N_cam,self.N_time* self.W * self.H, 1) or self.importance is None

        else:
            self.N_time=len(self.times)
            assert self.rays_rgbs.shape == (self.N_cam, self.N_time , self.W * self.H, 3)


    def __getitem__(self, idx):

        if self.split.startswith('train'):
            '''
            self.rays_rgbs.shape == (self.N_cam,self.N_time* self.W * self.H, 3)
            self.importance.shape == (self.N_cam,self.N_time* self.W * self.H, 1)
            self.times: [N_time] tensor
            
            '''
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_time': # randomly select across time

                cam_idxs = np.random.choice(self.N_cam, self.batch_size,p=None,replace=True)
                time_indices = np.random.choice(self.N_time, self.batch_size,p=None,replace=True)


            else:
                assert self.ray_sampling_strategy == 'same_time' # randomly select ONE time stamp
                cam_idxs = np.random.choice(self.N_cam, self.batch_size,p=None,replace=True)

                time_indices = np.random.choice(self.N_time,1)[0]
                #time_indices = time_indices*np.ones(self.batch_size).astype(np.int)

            times =self.times[cam_idxs,time_indices] # actually, for each camera it's identical

            ray_indices=np.random.choice(self.W*self.H,self.batch_size,p=None,replace=True)

            rgbs = self.rays_rgbs.view(self.N_cam, self.N_time, self.H * self.W, 3)[cam_idxs, time_indices, ray_indices]


            #print(f'times {times}')

            # randomly select pixels
            '''
            pixel indices is to get directions.
            '''
            pix_idxs = ray_indices


            #breakpoint()

            #print(f'im {cam_idxs.shape},pix{pix_idxs.shape},times{times.shape},rgb{rgbs.shape}')
            sample = {'img_idxs': cam_idxs, 'pix_idxs': pix_idxs,
                      'times':times,
                      'rgb': rgbs[:, :3]}
            #print(sample)

        else:


            '''
            bugs. need to stack N_CAM and N_time together...
            '''
            # test time
            '''
            self.rays_rgbs.shape == (self.N_cam, self.N_time , self.W * self.H, 3)
            '''


            sample = {'pose': self.poses[0], 'img_idxs': idx,
                      'times':torch.Tensor([self.times[idx]])
                      }
            if len(self.rays_rgbs)>0: # if ground truth available
                rgbs = self.rays_rgbs.view(-1,self.H*self.W, 3)[idx,:,:] # stack the different cams and different times
                #print(f'rgbs{self.rays_rgbs},{self.rays_rgbs.shape}')
                sample['rgb'] = rgbs

        return sample

    #def __len__(self):
    #    if self.split.startswith('train'):
        #    return self.N_cam*self.N_time* self.W * self.H
        #return self.N_time*self.N_cam
    def __len__(self):
        return self.N_time
