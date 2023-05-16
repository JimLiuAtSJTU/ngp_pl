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

        h,w=self.img_wh #using ngp ray direction characteristics

        self.directions = get_ray_directions(h, w, self.K)



