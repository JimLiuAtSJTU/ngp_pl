import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp
import numpy as np

from .rendering import NEAR_DISTANCE



class NGP_time_code_slim(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid'):
        super().__init__()

        self.rgb_act = rgb_act
        # scene bounding box
        self.scale = scale
        self.time_stamps = 300
        self.time_scale = 1
        '''
        assume time in [-1,1]
        '''
        self.t_center= torch.zeros(1)
        self.t_min= -torch.ones(1)*self.time_scale
        self.t_max= torch.ones(1)*self.time_scale

        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
                             torch.zeros(self.cascades * self.grid_size ** 3 // 8, dtype=torch.uint8))

        self.encoder_static = self.__get_hash_encoder(input_dims=3)
        self.encoder_dynamic = self.__get_hash_encoder(input_dims=3,config='xyz_dynamic')
        self.time_latent_code = self.__get_hash_encoder(input_dims=1, config='time_latent_code')
        self.xyzt_fusion_mlp=self.__get_fused_mlp(input_dims=64,output_dims=16)

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        flow_dim = 0
        self.rgb_net_static = self.__get_fused_mlp(input_dims=32, output_dims=3)
        self.rgb_net_dynamic = self.__get_fused_mlp(input_dims=32, output_dims=3 + 1 + flow_dim)  # rho for another dim

        if self.rgb_act == 'None':  # rgb_net output is log-radiance
            raise NotImplementedError
        print(f'time aware NGP model initialized')

    def __get_hash_encoder(self, input_dims=3, config='xyz_mlp'):
        # constants
        if config=='xyz_mlp':
            L = 16;
            F = 2;
            log2_T = 19;
            N_min = 16
            b = np.exp(np.log(2048 * self.scale / N_min) / (L - 1))

            return tcnn.NetworkWithInputEncoding(
                n_input_dims=input_dims, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        elif config=='time_latent_code':
            # out = 16*4=64dim or 8 * 8 =64dim
            L = 8;
            F = 4;     # time latent code length
            log2_T = 8; # 256 hash tables.
            N_min = 2 #   300 frames, each part = 50framse   total, 10s.
            highest_reso=self.time_stamps*0.666 # lower than the dimension
            b = np.exp(np.log(highest_reso * self.time_scale / N_min) / (L - 1))
            return tcnn.Encoding(
                n_input_dims=input_dims,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
            )


        elif config=='xyz_dynamic':
            L = 16;
            F = 2;
            log2_T = 19;
            N_min = 16
            b = np.exp(np.log(2048 * self.scale / N_min) / (L - 1))

            return tcnn.Encoding(
                n_input_dims=input_dims,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
            )

        else:
            raise NotImplementedError

    def __get_fused_mlp(self, input_dims=32, output_dims=3):

        return tcnn.Network(
            n_input_dims=input_dims, n_output_dims=output_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": self.rgb_act,
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )


    def blend_together(self, s_sigma, d_sigma, s_rgb, d_rgb, rho):
        '''

        SUDS blending.
        #https://arxiv.org/abs/2303.14536
        s_sigma,d_sigma: static, dynamic
        s_rgb,d_rgb: static, dynamic
        rho: shadow factor in [0,1]. consider using a sigmoid.
        '''
        sigma = s_sigma + d_sigma

        eps = 1e-6
        w_static = s_sigma / torch.clamp(sigma, min=eps)
        # print(f's_sigma{s_sigma}{s_sigma.shape}')
        # print(f'w_static{w_static}{w_static.shape}')
        # print(f's_rgb{s_rgb}{s_rgb.shape}')

        # print(f'rho{rho,}{rho.shape}')

        # unsqueeze
        rgb = (w_static * (1 - rho))[:, None] * s_rgb
        # print(f'rgb,{rgb.shape}')
        # print(f's_rgb,{s_rgb.shape}')

        rgb = rgb + (1 - w_static)[:, None] * d_rgb

        # return sigma,rgb
        '''

        return static for debug!
        '''
        return sigma, rgb, w_static


    def static_density(self, x, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        h = self.encoder_static(x)

        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def dynamic_density(self, x,t, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        x_encoding = self.encoder_dynamic(x)
        time_code=self.time_latent_code(t)

        h=self.xyzt_fusion_mlp(torch.cat([x_encoding,time_code],1))

        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """

        t = kwargs.get('times')
        try:
            assert t.shape[0] == x.shape[0]
        except AssertionError:
            # print(f'x,t',x,t)
            # print(f'x,t',x.shape,t.shape)

            assert t.shape[0] == 1

            t = t.expand(x.shape[0], 1)
        # except IndexError:
        #    assert isinstance(t,torch.Tensor)
        #    print(f't,{t},{type(t)}')

        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)

        sigma_static, h_static = self.static_density(x, return_feat=True)
        rgb_static = self.rgb_net_static(torch.cat([d, h_static], 1))

        sigma_dynamic, h_dyna = self.dynamic_density(x,t,  return_feat=True)


        #print(time_code.shape)
        #print(d.shape)
        #print(h_dyna.shape)

        rgb_dynamic = self.rgb_net_dynamic(torch.cat([d, h_dyna], 1))
        #exit(9)

        extra = {}
        sigma, rgb, weight = self.blend_together(s_sigma=sigma_static,
                                                 d_sigma=sigma_dynamic,
                                                 s_rgb=rgb_static,
                                                 d_rgb=rgb_dynamic[:, :-1],
                                                 rho=rgb_dynamic[:, -1])
        extra['static_weight'] = weight

        return sigma, rgb, extra

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
                coords2 = vren.morton3D_invert(indices2.int())
            else:
                indices2 = None
                coords2 = None

            # concatenate
            # cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            # torch.cuda.empty_cache()
            # print(f'coords{coords1},{coords1.shape},{coords2},{coords2.shape}')
            # print(f'indices{indices1},{indices1.shape},{indices2},{indices2.shape}')

            if indices2 is not None:
                cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            else:
                cells += [(None, None)]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=16 ** 3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a')  # (N_cams, 3, 3)
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i + chunk] / (self.grid_size - 1) * 2 - 1
                s = min(2 ** (c - 1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (uvd[:, 2] >= 0) & \
                           (uv[:, 0] >= 0) & (uv[:, 0] < img_wh[0]) & \
                           (uv[:, 1] >= 0) & (uv[:, 1] < img_wh[1])
                covered_by_cam = (uvd[:, 2] >= NEAR_DISTANCE) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i + chunk]] = \
                    count = covered_by_cam.sum(0) / N_cams

                too_near_to_cam = (uvd[:, 2] < NEAR_DISTANCE) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                self.density_grid[c, indices[i:i + chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        print(f'updating density grid, warmup={warmup}')

        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size ** 3 // 4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            if indices is None and coords is None:
                continue
            s = min(2 ** (c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords / (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.static_density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay ** (1 / self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid < 0,
                        self.density_grid,
                        torch.maximum(self.density_grid * decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid > 0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)