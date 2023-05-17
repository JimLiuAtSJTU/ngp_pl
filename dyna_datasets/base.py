from torch.utils.data import Dataset
import numpy as np
import torch
class BaseDataset(Dataset):
    """
    Define length and sampling method
    """
    def __init__(self, root_dir, split='train', downsample=1.0):
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return len(self.rays_rgbs)
        return max(self.times.shape)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            '''
            self.rays_rgbs.shape == (self.N_cam,self.N_time* self.W * self.H, 3)
            self.importance.shape == (self.N_cam,self.N_time* self.W * self.H, 1)
            '''
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images': # randomly select images
                cam_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image': # randomly select ONE image
                cam_idxs = np.random.choice(len(self.poses), 1)[0]

            # self.times is N_CAM, time_frames, 1
            t_indices=np.random.choice((self.times.shape[1]), 1) # randomly select ONE Time stamp
            times =self.times[cam_idxs,t_indices]

            #print(f'times {times}')

            # randomly select pixels
            pix_idxs = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
            rgbs = self.rays_rgbs[cam_idxs,t_indices, pix_idxs]

            #breakpoint()

            #print(f'im {cam_idxs.shape},pix{pix_idxs.shape},times{times.shape},rgb{rgbs.shape}')
            sample = {'img_idxs': cam_idxs, 'pix_idxs': pix_idxs,
                      'times':times,
                      'rgb': rgbs[:, :3]}
            #print(sample)

        else:
            # time stamp should match image stamp
            '''
            self.rays_rgbs.shape == (self.N_cam, self.N_time , self.W * self.H, 3)

            '''
            sample = {'pose': self.poses[idx], 'img_idxs': idx,
                      'times':torch.Tensor([self.times[idx]])
                      }
            if len(self.rays_rgbs)>0: # if ground truth available
                rgbs = self.rays_rgbs[idx]
                #print(f'rgbs{self.rays_rgbs},{self.rays_rgbs.shape}')
                sample['rgb'] = rgbs

        return sample

