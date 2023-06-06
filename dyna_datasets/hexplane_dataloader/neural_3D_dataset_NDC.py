import concurrent.futures
import copy
import gc
import glob
import os
import time

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from .ray_utils import get_ray_directions_blender, get_rays, ndc_rays_blender

def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0)) # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0) # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z)) # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x) # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def process_video(video_data_save, video_path, img_wh, downsample, transform):
    """
    Load video_path data to video_data_save tensor.
    """
    video_frames = cv2.VideoCapture(video_path)
    count = 0
    while video_frames.isOpened():
        ret, video_frame = video_frames.read()
        if ret:
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            video_frame = Image.fromarray(video_frame)
            if downsample != 1.0:
                img = video_frame.resize(img_wh, Image.LANCZOS)
            img = transform(img) # C, H, W
            #video_data_save[count] = img.permute(2, 1, 0) #if you need to locate pixel
            video_data_save[count] = img.view(3, -1).permute(1, 0) # H*W, C=3
            count += 1
        else:
            break
    video_frames.release()
    print(f"Video {video_path} processed.")
    return None


# define a function to process all videos
def process_videos(videos, skip_index, img_wh, downsample, transform, num_workers=1):
    """
    A multi-threaded function to load all videos fastly and memory-efficiently.
    To save memory, we pre-allocate a tensor to store all the images and spawn multi-threads to load the images into this tensor.
    """
    #all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[0],img_wh[1], 3) #if you need to locate pixel
    all_imgs = torch.zeros(len(videos) - 1, 300, img_wh[-1] * img_wh[-2], 3) # N_CAM, T, H*W, C
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        # start a thread for each video
        current_index = 0
        futures = []
        for index, video_path in enumerate(videos):
            # skip the video with skip_index (eval video)
            if index == skip_index:
                continue
            else:
                future = executor.submit(
                    process_video,
                    all_imgs[current_index],
                    video_path,
                    img_wh,
                    downsample,
                    transform,
                )
                futures.append(future)
                current_index += 1
    return all_imgs


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)


class Neural3D_NDC_Dataset(Dataset):
    def __init__(
        self,
        datadir,
        split="train",
        downsample=(1/(2704/2028)),
        is_stack=True,
        cal_fine_bbox=False,
        N_vis=-1,
        time_scale=1.0,
        scene_bbox_min=[-1.0, -1.0, -1.0],
        scene_bbox_max=[1.0, 1.0, 1.0],
        N_random_pose=1000,
        bd_factor=0.75,
        eval_step=1,
        eval_index=0,
        sphere_scale=1.0,
    ):
        self.img_wh = (
        #    int(1024 / downsample),
        #    int(768 / downsample),
            int(2704 / downsample),
            int(2028 / downsample),

        )  # According to the neural 3D paper, the default resolution is 1024x768
        self.root_dir = datadir
        self.split = split
        self.downsample = 2704 / self.img_wh[0]
        self.is_stack = is_stack
        self.N_vis = N_vis
        self.time_scale = time_scale
        self.scene_bbox = torch.tensor([scene_bbox_min, scene_bbox_max])

        self.world_bound_scale = 1.1
        self.bd_factor = bd_factor
        self.eval_step = eval_step
        self.eval_index = eval_index
        self.blender2opencv = np.eye(4)
        self.transform = T.ToTensor()

        self.near = 0.0
        self.far = 1.0
        self.near_far = [self.near, self.far]  # NDC near far is [0, 1.0]
        self.white_bg = False
        self.ndc_ray = True
        self.depth_data = False

        self.load_meta()
        print("meta data loaded")

    def load_meta(self):
        """
        Load meta data from the dataset.
        """
        # Read poses and video file paths.
        poses_arr = np.load(os.path.join(self.root_dir, "poses_bounds.npy"))
        poses = poses_arr[:, :-2].reshape([-1, 3, 5])  # (N_cams, 3, 5)
        self.near_fars = poses_arr[:, -2:]
        videos = glob.glob(os.path.join(self.root_dir, "cam*.mp4"))
        videos = sorted(videos)
        assert len(videos) == poses_arr.shape[0]
        #visualize_poses(poses)

        H, W, focal = poses[0, :, -1]



        assert self.img_wh[0]*H == self.img_wh[1]* W # self.img_wh: w,h = W,H

        '''
        scale focal
        '''



        self.focal = np.array([focal, focal])
        self.focal *= self.img_wh[0]/W

        #poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        #visualize_poses(poses)

        # torch-ngp
        # inversion of this: https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/llff/poses/pose_utils.py#L51
        '''
        IMPORTANT!
        DIFFERENT!!!
        '''

        poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:3], poses[..., 3:4]], -1)  # (N, 3, 4)

        visualize_poses(poses)


        poses, pose_avg = center_poses(
            poses, self.blender2opencv
        )  # Re-center poses so that the average is near the center.
        near_original0=np.linalg.norm(poses[..., 3], axis=-1).max()
        near_original = self.near_fars.min()


        '''
        see
        https://github.com/bmild/nerf/issues/34
        for more details.
        
        correcting the poses is not easy.
        '''

        #near_original = self.near_fars[:,0].max()
        scale_factor = near_original * 2
        self.near_fars /= (
            scale_factor  # rescale nearest plane so that it is at z = 4/3.
        )

        '''
        
        
        
        should change the poses coordinates.
        s.t. the most of the scene objects in 'centered' bouding box.
        
        consider: 
        1. choose a direction vector  d=(dx,dy,dz), s.t. sum <d, d_i> is the least
        actually select the mean dx,dy,dz is ok.
        
        2. choose a distance, and set the offset, and translate the poses.
        
        find a point P on the direction, s.t. the summed  P-d_i  distance  
        
        is minimised. just by computing and selecting argmin is OK.
        
        consider using the visualization function for help.
        
        '''
        poses[..., 3] /= scale_factor


        '''
        the given LLFF poses is not working. the cameras are centered at (0,0,0).
        this will lead to some problems for ngp-ray-marching.
        
        need to correct poses, s.t. the scene is centered at (0,0,0)
        
        simply by setting the camera ray intersection to (0,0,0) is not enough
        
        because camera ray intersection may not be the center.
        
        '''
        poses=correct_poses(poses,distance_factor=1)




        visualize_poses(poses)
        # Sample N_views poses for validation - NeRF-like camera trajectory.
        N_views = 120

        all_indx=range(poses.shape[0])
        val_indx=[self.eval_index]
        train_indx=list(set(all_indx)-set(val_indx))
        self.val_poses = get_spiral(poses, self.near_fars, N_views=N_views)

        W, H = self.img_wh

        if self.split=='train':
            self.poses=torch.FloatTensor(poses[train_indx])
        else:
            self.poses=torch.FloatTensor(poses[val_indx])

        K = np.float32([[self.focal[0], 0, W/2],
                        [0, self.focal[1], H/2],
                        [0,  0,   1]])
        #print(np.linalg.cond(K))
        self.K = torch.FloatTensor(K)


        self.directions = torch.tensor(
            get_ray_directions_blender(H, W, self.focal)
        )  # (H, W, 3)
        max_time_count = 300

        if self.split == "train":
            # Loading all videos from this dataset requires around 50GB memory, and stack them into a tensor requires another 50GB.
            # To save memory, we allocate a large tensor and load videos into it instead of using torch.stack/cat operations.
            all_times = []
            all_rays = []

            for index in range(0, len(videos)):
                if (
                    index == self.eval_index
                ):  # the eval_index(0 as default) is the evaluation one. We skip evaluation cameras.
                    continue

                video_times = torch.tensor([i / (max_time_count ) for i in range(max_time_count)])
                all_times += [video_times]

                rays_o, rays_d = get_rays(
                    self.directions, torch.FloatTensor(poses[index])
                )  # both (h*w, 3)
                rays_o, rays_d = ndc_rays_blender(H, W, focal, 1.0, rays_o, rays_d)
                all_rays += [torch.cat([rays_o, rays_d], 1)]
                print(f"video {index} is loaded")
                gc.collect()

            # load all video images
            all_imgs = process_videos(
                videos,
                self.eval_index,
                self.img_wh,
                self.downsample,
                self.transform,
                num_workers=8,
            )
            all_times = torch.stack(all_times, 0)
            all_rays = torch.stack(all_rays, 0)
            print("stack performed")

            # if you need to locate pixel
            #N_cam, N_time, H_,W_, C = all_imgs.shape
            #N_rays=H_*W_
            #self.image_stride = (H,W)
            N_cam, N_time, N_rays, C = all_imgs.shape # N_CAM, T, H*W, C
            importances_=torch.zeros(N_cam, N_time, N_rays,1)
            '''
            IF IMPORTANCE should be calculated, may calculate it in the multi-processing 
            '''
            # use gpu to compute importance. hopefully it would be faster

            t0=time.time()

            sums=[]
            device_='cpu' if torch.cuda.is_available() else "cpu"
            for i in range(N_cam):
                # importances for out-of mean value
                current_video=all_imgs[i].to(device_)

                average_=torch.mean(current_video,dim=0)

                diff= torch.abs(average_-current_video)


                importances_[i]= torch.sum(diff,dim=-1).unsqueeze(-1).to('cpu')
                #sums.append(torch.sum(importances_[i]))

            for i in range(N_cam):
                # importances for difference between frames

                current_video = all_imgs[i].to(device_)

                frames=np.array(range(1,current_video.shape[0])).astype(int)


                difference=torch.abs(current_video[frames] - current_video[frames-1]) # frames -1 , H,W,3


                #pad the initial frame difference
                importances_[i,1:] += torch.sum(difference, dim=-1).unsqueeze(-1).to('cpu')
                importances_[i,0] += torch.sum(difference[0], dim=-1).unsqueeze(-1).to('cpu')

                importances_[i] = importances_[i]/torch.sum(importances_[i])
                #sums.append(torch.sum(importances_[i]))
            #std_mean = torch.std_mean(importances_[:,1:,:])
            #importances_[:,0,:] = std_mean[0]/4 + std_mean[1] # set the initial frame > average importance , to assure more importance on static scene
            t1=time.time()

            print(f'importance calculated. time elapse={t1-t0}')

            self.image_stride = N_rays
            self.cam_number = N_cam
            self.time_number = N_time
            self.all_rgbs = all_imgs.view(N_cam,N_time*N_rays,C)#
            self.all_times = all_times.view(N_cam, N_time, 1)
            self.all_rays = all_rays.reshape(N_cam, N_rays, 6)
            self.all_importances=importances_.view(N_cam,N_time,N_rays,1)#

            # using pytorch to perform importance sampling seems to have some problems...
            # consider change to numpy and use double precision.
            # distri = torch.distributions.Categorical(torch.squeeze(self.all_importances)/torch.sum(self.all_importances))
            # sample = distri.sample([8192])
            self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)
            self.global_mean_rgb = torch.mean(all_imgs, dim=1)
        else:
            assert isinstance(self.eval_index,int) #only one eval index
            N_cam=1
            index = self.eval_index
            video_imgs = []
            video_frames = cv2.VideoCapture(videos[index])

            time_cnt=0
            while video_frames.isOpened():
                ret, video_frame = video_frames.read()
                if ret:
                    video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                    video_frame = Image.fromarray(video_frame)
                    if self.downsample != 1.0:
                        img = video_frame.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img) # H,W,3
                    #video_imgs += [img.permute(2, 1, 0)] #if you need to locate pixel
                    video_imgs += [img.view(3, -1).permute(1, 0)] # H*W, 3
                    time_cnt += 1
                    if time_cnt>=max_time_count:
                        break

                else:
                    break
            video_imgs = torch.stack(video_imgs, 0) #
            video_times = torch.tensor(
                [i / (len(video_imgs) ) for i in range(len(video_imgs))]
            )
            video_imgs = video_imgs[0 :: self.eval_step]
            video_times = video_times[0 :: self.eval_step]
            rays_o, rays_d = get_rays(
                self.directions, torch.FloatTensor(poses[index])
            )  # both (h*w, 3)
            rays_o, rays_d = ndc_rays_blender(H, W, focal, 1.0, rays_o, rays_d)
            all_rays = torch.cat([rays_o, rays_d], 1)
            gc.collect()
            #N_time,  H,W, C = video_imgs.shape #if you need to locate pixel
            #self.image_stride = (H,W) #if you need to locate pixel
            N_time, N_rays, C = video_imgs.shape
            self.image_stride = N_rays
            self.time_number = N_time
            #self.all_rgbs = video_imgs #if you need to locate pixel
            self.all_rgbs = video_imgs.view(N_cam,-1, N_rays, 3) # N_val=1, N_time, N_rays, 3
            self.all_rays = all_rays.view(N_cam,N_rays,6)
            self.all_times = video_times
            #self.all_rgbs = self.all_rgbs.view(
            #    -1, *self.img_wh[::-1], 3
            #)  # (len(self.meta['frames]),h,w,3)
            self.all_times = self.time_scale * (self.all_times * 2.0 - 1.0)

    def __len__(self):
        if self.split == "train" and self.is_stack is True:
            return self.cam_number * self.time_number
        else:
            return len(self.all_rgbs)

    def __getitem__(self, idx):
        if self.split == "train":  # use data in the buffers
            if self.is_stack:
                cam_idx = idx // self.time_number
                time_idx = idx % self.time_number
                sample = {
                    "rays": self.all_rays[cam_idx],
                    "rgbs": self.all_rgbs[cam_idx, time_idx],
                    "time": self.all_times[cam_idx, time_idx]
                    * torch.ones_like(self.all_rays[cam_idx][:, 0:1]),
                }

            else:
                sample = {
                    "rays": self.all_rays[
                        idx // (self.time_number * self.image_stride),
                        idx % (self.image_stride),
                    ],
                    "rgbs": self.all_rgbs[idx],
                    "time": self.all_times[
                        idx // (self.time_number * self.image_stride),
                        idx
                        % (self.time_number * self.image_stride)
                        // self.image_stride,
                    ]
                    * torch.ones_like(self.all_rgbs[idx][:, 0:1]),
                }

        else:  # create data for each image separately
            if self.is_stack:
                sample = {
                    "rays": self.all_rays,
                    "rgbs": self.all_rgbs[idx],
                    "time": self.all_times[idx]
                    * torch.ones_like(self.all_rays[:, 0:1]),
                }

            else:
                sample = {
                    "rays": self.all_rays[idx % self.image_stride],
                    "rgbs": self.all_rgbs[idx],
                    "time": self.all_times[idx // self.image_stride]
                    * torch.ones_like(self.all_rays[:, 0:1]),
                }

        return sample

    def get_val_pose(self):
        render_poses = self.val_poses
        render_times = torch.linspace(0.0, 1.0, render_poses.shape[0]) * 2.0 - 1.0
        return render_poses, self.time_scale * render_times

    def get_val_rays(self):
        val_poses, val_times = self.get_val_pose()  # get valitdation poses and times
        rays_all = []  # initialize list to store [rays_o, rays_d]

        for i in range(val_poses.shape[0]):
            c2w = torch.FloatTensor(val_poses[i])
            rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
            if self.ndc_ray:
                W, H = self.img_wh
                rays_o, rays_d = ndc_rays_blender(
                    H, W, self.focal[0], 1.0, rays_o, rays_d
                )
            rays = torch.cat([rays_o, rays_d], 1)  # (h*w, 6)
            rays_all.append(rays)
        return rays_all, torch.FloatTensor(val_times)

import trimesh
import warnings

visual_=False

def visualize_poses(poses, size=0.1):
    # poses shoould be : [B, 4, 4]
    size_=poses.shape[1]
    if size_==3:
        last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N, 1, 4)
        poses = np.concatenate([poses, last_row], axis=1)  # (N, 4, 4)

    if not visual_:
        warnings.warn('visualize is disabled.')
        return
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i,pose in enumerate(poses):
        print(f'pose {i},={pose}')
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def correct_poses(poses,size=0.1,distance_factor=0):
    # poses shoould be : [B, 4, 4]
    size_=poses.shape[1]
    if size_==3:
        last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N, 1, 4)
        poses = np.concatenate([poses, last_row], axis=1)  # (N, 4, 4)

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(extents=(2, 2, 2)).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]





    #dir_sum=np.sum(directions,axis=0)
    for i,pose in enumerate(poses):
        print(f'pose {i},={pose}')
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)

        #dir_sum = dir_sum + dir
        #directions[i]=dir
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(copy.deepcopy(segs))

    if visual_:trimesh.Scene(objects).show()

    '''
    mean directions
    '''
    directions=np.zeros_like((poses[:,:3,3]))

    dir_sum=np.zeros(3)

    for i,pose in enumerate(poses):
        print(f'pose {i},={pose}')
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)

        tmp_dir=copy.deepcopy(dir)
        dir_sum = dir_sum + tmp_dir
        directions[i]= tmp_dir
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a], [pos, o]])
        segs = trimesh.load_path(segs)
        objects.append(copy.deepcopy(segs))

    if visual_:trimesh.Scene(objects).show()



    mean_dir = dir_sum / (np.linalg.norm(dir_sum) + 1e-8)

    mean_offset = np.mean(poses[:,:3,3],axis=0)



    '''
    we need to search for the mi
    '''

    origin=np.array([0,0,0])

    visual_mean_dir=np.array([[mean_offset,mean_offset+mean_dir*6]])

    visual_mean_dir = trimesh.load_path(visual_mean_dir)
    objects.append(copy.deepcopy(visual_mean_dir))

    if visual_:trimesh.Scene(objects).show()


    points_num=5000

    coord_poins=np.linspace(-30,30,points_num,endpoint=False)[:,None] * mean_dir

    lengths_=np.zeros(points_num)
    for i in range(points_num):
        point_=coord_poins[i]

        #compute projection length
        #https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points

        p1=poses[:,:3,3]
        p2=p1+directions
        p3=point_

        d = np.cross(p2 - p1, p3-p1) / np.linalg.norm(p2-p1,axis=-1 )[:,None] #
        projection_length= sum( abs( np.linalg.norm(d,axis=-1)))

        lengths_[i]=projection_length

    indx=np.argmin(lengths_)

    desired_offset=coord_poins[indx]

    # poses is concatenated to N,4,4
    # return should be N,3,4
    corrected_poses=poses[:,:3,:]

    corrected_poses[:, :3, 3] -= desired_offset*(1+distance_factor)  # N, 3, 4



    #visual_mean_dir = trimesh.load_path(visual_mean_dir)
    #objects.append(visual_mean_dir)

    #trimesh.Scene(objects).show()


    return corrected_poses
