import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp, TruncExp_2
import numpy as np
import torch.nn.functional as F
from .rendering import NEAR_DISTANCE
from kornia.utils.grid import create_meshgrid3d
from .debug_utils import nan_check


class NGP_time_code_simple(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid',static_only=False):
        super().__init__()
        self.static_only=static_only
        self.rgb_act = rgb_act
        # scene bounding box
        self.scale = scale
        self.time_stamps = 300
        self.time_scale = 1
        '''
        assume time in [-1,1]
        '''
        self.t_min = - self.time_scale  # -1 as default
        self.t_max = self.time_scale  # 1 as default

        self.time_grid_resolution = 1  # may be fine-tuned
        self.t_center = torch.zeros(1)
        # self.t_min= -torch.ones(1)*self.time_scale
        # self.t_max= torch.ones(1)*self.time_scale

        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)  # int
        self.grid_size = 128
        self.register_buffer('density_bitfield',
                             torch.zeros(self.time_grid_resolution, self.cascades * self.grid_size ** 3 // 8,
                                         dtype=torch.uint8))
        '''
        divide xyz and xyz_fusion will not lead to performance increase.
        maybe it's because of the complex computation graph.
        to acceletate, we may need to tune the updating strategy of density grid
        '''
        self.encoder_static = self.__get_hash_encoder(input_dims=3)
        self.encoder_dynamic = self.__get_hash_encoder(input_dims=3, config='xyz_dynamic')
        self.time_latent_code = self.__get_hash_encoder(input_dims=1, config='time_latent_code')
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        sigma_factor_dim = 0

        self.xyz_t_fusion_mlp = tcnn.Network(
            n_input_dims=64, n_output_dims=48,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )
        self.rgb_net_static = self.__get_rgb_mlp(input_dims=32, output_dims=3)
        self.rgb_net_dynamic = self.__get_rgb_mlp(input_dims=64,
                                                  output_dims=3 + 1 + sigma_factor_dim)  # rho for another dim
        self.fine_tune=False
        if self.rgb_act == 'None':  # rgb_net output is log-radiance
            raise NotImplementedError

        self.init_density_grids()

        self.background_rgb_mlp = tcnn.NetworkWithInputEncoding(
            n_input_dims=7, n_output_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": 'SphericalHarmonics',
                        "degree": 3,
                    },
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    {
                        "n_dims_to_encode": 1,
                        "otype": "Frequency",
                        "n_frequencies": 8,
                    }

                ]
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,  # maybe only one layer is enough?
            }
        )

        print(f'time aware NGP model initialized')

    def init_density_grids(self):
        G = self.grid_size
        self.register_buffer('density_grid',
                             torch.zeros(self.time_grid_resolution, self.cascades, G ** 3))

        tmp = create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3)
        size_ = tmp.shape[0]
        self.register_buffer('grid_coords', torch.zeros(self.time_grid_resolution, size_, 3, dtype=torch.int32))
        for i in range(self.time_grid_resolution):
            self.grid_coords[i] = tmp[:, :]

    def __get_hash_encoder(self, input_dims=3, config='xyz_mlp'):
        # constants
        if config == 'xyz_mlp':
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
                    "n_hidden_layers": 1,  # maybe only one layer is enough?
                }
            )
        elif config == 'time_latent_code':

            L = 2;
            F = 20;  # 40 dim
            log2_T = 9;  # 256 hash tables.
            N_min = 120  # 300 frames, each part = 50framse   total, 10s.
            highest_reso = self.time_stamps * 0.666  # lower than the dimension
            b = np.exp(np.log(highest_reso * self.time_scale / N_min) / (L - 1))
            '''
            '''
            return tcnn.Encoding(
                n_input_dims=1,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                }
            )

        elif config == 'xyz_dynamic':

            L = 12;
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
                }
            )
        else:
            raise NotImplementedError

    def __get_rgb_mlp(self, input_dims=32, output_dims=3):

        return tcnn.Network(
            n_input_dims=input_dims, n_output_dims=output_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
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
        sigma = s_sigma + d_sigma * (1 - rho)

        eps = 1e-6
        w_static = s_sigma / torch.clamp(sigma, min=eps)
        # print(f's_sigma{s_sigma}{s_sigma.shape}')
        # print(f'w_static{w_static}{w_static.shape}')
        # print(f's_rgb{s_rgb}{s_rgb.shape}')

        # print(f'rho{rho,}{rho.shape}')

        # unsqueeze
        rgb = (w_static)[:, None] * s_rgb
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

        nan_check(x)
        if self.fine_tune:
            with torch.no_grad():
                h = self.encoder_static(x)
        else:
            h = self.encoder_static(x)

        nan_check(h)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def dynamic_density(self, x, t, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        t = (t - self.t_min) / (self.t_max - self.t_min)

        if self.fine_tune:
            with torch.no_grad():
                static_code = self.encoder_dynamic(x)
        else:
            static_code = self.encoder_dynamic(x)

            # # https://github.com/NVlabs/tiny-cuda-nn/issues/286
        # tcnn supports inputs within [0,1]

        nan_check(x)
        nan_check(t)
        assert x.shape[0] == t.shape[0]

        try:
            if self.fine_tune:
                with torch.no_grad():
                    time_code = self.time_latent_code(t)
            else:
                time_code = self.time_latent_code(t)

        except RuntimeError:
            print(f't.shape,{t.shape}')
            print(t)
            exit(1)

        nan_check(time_code)
        h = self.xyz_t_fusion_mlp(torch.cat([static_code, time_code], 1))
        #h = torch.cat([h,static_code,time_code],-1)
        nan_check(h)
        sigmas = TruncExp_2.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def background_field(self, rayso, raysd,t):
        return self.background_rgb_mlp(torch.cat([rayso,raysd,t],-1))

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

        if kwargs.get('background_field', False):
            # compute backgournd field
            rays_o, rays_d = kwargs.get('rays_o'), kwargs.get('rays_d')
            try:
                assert rays_d.shape[0]==rays_o.shape[0]==t.shape[0]

            except:
                print(rays_o.shape)
                print(rays_d.shape)
                print(t.shape)
                exit(1)
            t = (t - self.t_min) / (self.t_max - self.t_min)

            return self.background_field(rays_o, rays_d,t)


        try:
            assert t.shape[0] == x.shape[0]
        except AssertionError:
            # print(f'x,t',x,t)
            # print(f'x,t',x.shape,t.shape)

            assert t.shape[0] == 1

            t = t.expand(x.shape[0], 1)
        nan_check(t)
        try:
            assert x.shape[0] > 0
        except:
            print(x)
            print(x.shape)
            exit(-1)
        nan_check(x)
        nan_check(d)


        # except IndexError:
        #    assert isinstance(t,torch.Tensor)
        #    print(f't,{t},{type(t)}')

        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)

        sigma_dynamic, h_dyna = self.dynamic_density(x, t, return_feat=True)
        nan_check(sigma_dynamic)
        rgb_dynamic = self.rgb_net_dynamic(torch.cat([d, h_dyna], 1))
        nan_check(rgb_dynamic)

        sigma_static=torch.zeros_like(sigma_dynamic)
        rgb_static=torch.zeros_like(rgb_dynamic[:, :-1])
    

        # print(time_code.shape)
        # print(d.shape)
        # print(h_dyna.shape)
        # exit(9)
        extra = {
            'rgb_dynamic': rgb_dynamic[:, :3],
            'sigma_dynamic': sigma_dynamic,
        }
        sigma, rgb, weight = self.blend_together(s_sigma=sigma_static,
                                                 d_sigma=sigma_dynamic,
                                                 s_rgb=rgb_static,
                                                 d_rgb=rgb_dynamic[:, :-1],
                                                 rho=rgb_dynamic[:, -1])
        extra['static_weight'] = weight
        extra['static_weight_average'] = weight.detach().mean()

        return sigma, rgb, extra

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords[0]).long()
        cells = [(indices, self.grid_coords[0])] * self.cascades

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
        t_cells = []
        for t in range(self.time_grid_resolution):
            cells = []
            for c in range(self.cascades):
                # uniform cells
                coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                        device=self.density_grid[t].device)
                indices1 = vren.morton3D(coords1).long()
                # occupied cells
                indices2 = torch.nonzero(self.density_grid[t, c] > density_threshold)[:, 0]
                if len(indices2) > 0:
                    rand_idx = torch.randint(len(indices2), (M,),
                                             device=self.density_grid[t].device)
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
            t_cells.append(cells)
        return t_cells

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
        self.count_grid = torch.zeros_like(self.density_grid[0])
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
                    count = covered_by_cam.sum(0)/ N_cams

                too_near_to_cam = (uvd[:, 2] < NEAR_DISTANCE) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                self.density_grid[:, c, indices[i:i + chunk]] = \
                    torch.where(valid_mask, 0., -1.)  # same for all time stamps
    # TODO: consider move updating density grid to forward region to perform AMP

    '''
    updating desntity_grid_should be in float32 operation, this may lead to performance issue.
    consider warp this operation in the forward function, to let lightning module call mixed precision plugin.
    may bring some performance bonus.
    '''
    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        # print(f'updating density grid, warmup={warmup}')

        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            t_cells = self.sample_uniform_and_occupied_cells(self.grid_size ** 3 // 4,
                                                             density_threshold)
        for t_ in range(self.time_grid_resolution):
            if not warmup:
                cells = t_cells[t_]
            density_grid_tmp = torch.zeros_like(self.density_grid[t_])

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
                '''
                according to the blendring together algorithm in Neural Scene Flow Fields
                we should add the static and dynamic density together.
                '''
                '''
                generate uniform random t in the range
                '''
                t_interval = self.t_max - self.t_min
                t_start_ = t_interval * (t_ / self.time_grid_resolution) + self.t_min
                t_end = t_interval * ((t_ + 1) / self.time_grid_resolution) + self.t_min
                rand_t = torch.rand_like(xyzs_w[:, 0:1]) * (t_end - t_start_) + t_end

                if self.static_only:
                    rand_t=torch.zeros_like(rand_t)
                density_grid_tmp[c, indices] = self.dynamic_density(xyzs_w, rand_t,  return_feat=False).float()
            # TODO: consider the decay.
            '''
            important!!!
            the 0.95 factor of decay may be not suitable for dynamic setting.
            0.95 is very good for static setting.
            '''
            if erode:
                # My own logic. decay more the cells that are visible to few cameras
                decay = torch.clamp(decay ** (1 / self.count_grid), 0.1, 0.95)
            self.density_grid[t_] = \
                torch.where(self.density_grid[t_] < 0,
                            self.density_grid[t_],
                            torch.maximum(self.density_grid[t_] * decay, density_grid_tmp))
            # print(2)
            mean_density = self.density_grid[t_][self.density_grid[t_] > 0].mean().item()
            # print(3)
            vren.packbits(self.density_grid[t_], min(mean_density, density_threshold),
                          self.density_bitfield[t_])

    def get_t_grid_indices(self, batched_time_stamps):
        t_min = - self.time_scale  # -1 as default
        t_max = self.time_scale  # 1 as default

        diff = (batched_time_stamps - t_min) / (t_max - t_min)  # compress into [0,1) to avoid coordinates overflow
        diff_indx = torch.floor(diff * self.time_grid_resolution)  # [0, t_resolution)
        diff_indx = diff_indx.int()
        return torch.clamp(diff_indx, 0, self.time_grid_resolution - 1)


class Naive_4DNGP(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid', static_only=False):
        super().__init__()
        self.static_only = static_only
        self.rgb_act = rgb_act
        # scene bounding box
        self.scale = scale
        self.time_stamps = 300
        self.time_scale = 1
        '''
        assume time in [-1,1]
        '''
        self.t_min = - self.time_scale  # -1 as default
        self.t_max = self.time_scale  # 1 as default

        self.time_grid_resolution = 1  # may be fine-tuned
        self.t_center = torch.zeros(1)
        # self.t_min= -torch.ones(1)*self.time_scale
        # self.t_max= torch.ones(1)*self.time_scale

        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)  # int
        self.grid_size = 128
        self.register_buffer('density_bitfield',
                             torch.zeros(self.time_grid_resolution, self.cascades * self.grid_size ** 3 // 8,
                                         dtype=torch.uint8))
        '''
        divide xyz and xyz_fusion will not lead to performance increase.
        maybe it's because of the complex computation graph.
        to acceletate, we may need to tune the updating strategy of density grid
        '''
        self.encoder_static = self.__get_hash_encoder(input_dims=3)
        self.encoder_dynamic = self.__get_hash_encoder(input_dims=3, config='xyz_dynamic')
        self.time_latent_code = self.__get_hash_encoder(input_dims=1, config='time_latent_code')
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )
        sigma_factor_dim = 0

        L = 16;
        F = 2;
        log2_T = 19;
        N_min = 16
        b = np.exp(np.log(2048  / N_min) / (L - 1))

        self.xyzt_4dmodel = tcnn.NetworkWithInputEncoding(
            n_input_dims=4, n_output_dims=48,
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
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )
        self.rgb_net_static = self.__get_rgb_mlp(input_dims=32, output_dims=3)
        self.rgb_net_dynamic = self.__get_rgb_mlp(input_dims=64,
                                                  output_dims=3 + 0 + sigma_factor_dim)  # rho for another dim
        self.fine_tune = False
        if self.rgb_act == 'None':  # rgb_net output is log-radiance
            raise NotImplementedError

        self.init_density_grids()

        self.background_rgb_mlp = tcnn.NetworkWithInputEncoding(
            n_input_dims=7, n_output_dims=3,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3,
                        "otype": 'SphericalHarmonics',
                        "degree": 3,
                    },
                    {
                        "n_dims_to_encode": 3,
                        "otype": "SphericalHarmonics",
                        "degree": 4,
                    },
                    {
                        "n_dims_to_encode": 1,
                        "otype": "Frequency",
                        "n_frequencies": 8,
                    }

                ]
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,  # maybe only one layer is enough?
            }
        )

        print(f'Naive 4D NGP')

    def init_density_grids(self):
        G = self.grid_size
        self.register_buffer('density_grid',
                             torch.zeros(self.time_grid_resolution, self.cascades, G ** 3))

        tmp = create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3)
        size_ = tmp.shape[0]
        self.register_buffer('grid_coords', torch.zeros(self.time_grid_resolution, size_, 3, dtype=torch.int32))
        for i in range(self.time_grid_resolution):
            self.grid_coords[i] = tmp[:, :]

    def __get_hash_encoder(self, input_dims=3, config='xyz_mlp'):
        # constants
        if config == 'xyz_mlp':
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
                    "n_hidden_layers": 1,  # maybe only one layer is enough?
                }
            )
        elif config == 'time_latent_code':

            L = 2;
            F = 20;  # 40 dim
            log2_T = 9;  # 256 hash tables.
            N_min = 120  # 300 frames, each part = 50framse   total, 10s.
            highest_reso = self.time_stamps * 0.666  # lower than the dimension
            b = np.exp(np.log(highest_reso * self.time_scale / N_min) / (L - 1))
            '''
            '''
            return tcnn.Encoding(
                n_input_dims=1,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                }
            )

        elif config == 'xyz_dynamic':

            L = 12;
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
                }
            )
        else:
            raise NotImplementedError

    def __get_rgb_mlp(self, input_dims=32, output_dims=3):

        return tcnn.Network(
            n_input_dims=input_dims, n_output_dims=output_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "LeakyReLU",
                "output_activation": self.rgb_act,
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )


    def dynamic_density(self, x, t, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        t = (t - self.t_min) / (self.t_max - self.t_min)

        if self.fine_tune:
            with torch.no_grad():
                static_code = self.encoder_dynamic(x)
        else:
            static_code = self.encoder_dynamic(x)

            # # https://github.com/NVlabs/tiny-cuda-nn/issues/286
        # tcnn supports inputs within [0,1]


        h = self.xyzt_4dmodel(torch.cat([x, t]
                                        , 1))
        # h = torch.cat([h,static_code,time_code],-1)
        nan_check(h)
        sigmas = TruncExp_2.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def background_field(self, rayso, raysd, t):
        return self.background_rgb_mlp(torch.cat([rayso, raysd, t], -1))


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

        if kwargs.get('background_field', False):
            # compute backgournd field
            rays_o, rays_d = kwargs.get('rays_o'), kwargs.get('rays_d')
            try:
                assert rays_d.shape[0]==rays_o.shape[0]==t.shape[0]

            except:
                print(rays_o.shape)
                print(rays_d.shape)
                print(t.shape)
                exit(1)
            t = (t - self.t_min) / (self.t_max - self.t_min)

            return self.background_field(rays_o, rays_d,t)


        try:
            assert t.shape[0] == x.shape[0]
        except AssertionError:
            # print(f'x,t',x,t)
            # print(f'x,t',x.shape,t.shape)

            assert t.shape[0] == 1

            t = t.expand(x.shape[0], 1)
        nan_check(t)

        assert x.shape[0] > 0
        nan_check(x)
        nan_check(d)


        # except IndexError:
        #    assert isinstance(t,torch.Tensor)
        #    print(f't,{t},{type(t)}')

        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)

        # use exp as activation, initialized to be 1

        sigma_dynamic, h_dyna = self.dynamic_density(x, t, return_feat=True)
        nan_check(sigma_dynamic)

        # print(time_code.shape)
        # print(d.shape)
        # print(h_dyna.shape)
        # exit(9)
        rgb_dynamic = self.rgb_net_dynamic(torch.cat([d, h_dyna], 1))
        nan_check(rgb_dynamic)
        extra = {
            'rgb_dynamic': rgb_dynamic[:, :3],
            'sigma_dynamic': sigma_dynamic,
        }

        weight=torch.ones_like(sigma_dynamic)
        extra['static_weight'] = weight
        extra['static_weight_average'] = weight.mean()

        return sigma_dynamic, rgb_dynamic, extra



    def blend_together(self, s_sigma, d_sigma, s_rgb, d_rgb, rho):
        '''

        SUDS blending.
        #https://arxiv.org/abs/2303.14536
        s_sigma,d_sigma: static, dynamic
        s_rgb,d_rgb: static, dynamic
        rho: shadow factor in [0,1]. consider using a sigmoid.
        '''
        sigma = s_sigma + d_sigma * (1 - rho)

        eps = 1e-6
        w_static = s_sigma / torch.clamp(sigma, min=eps)
        # print(f's_sigma{s_sigma}{s_sigma.shape}')
        # print(f'w_static{w_static}{w_static.shape}')
        # print(f's_rgb{s_rgb}{s_rgb.shape}')

        # print(f'rho{rho,}{rho.shape}')

        # unsqueeze
        rgb = (w_static)[:, None] * s_rgb
        # print(f'rgb,{rgb.shape}')
        # print(f's_rgb,{s_rgb.shape}')

        rgb = rgb + (1 - w_static)[:, None] * d_rgb

        # return sigma,rgb
        '''

        return static for debug!
        '''
        return sigma, rgb, w_static


    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.

        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords[0]).long()
        cells = [(indices, self.grid_coords[0])] * self.cascades

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
        t_cells = []
        for t in range(self.time_grid_resolution):
            cells = []
            for c in range(self.cascades):
                # uniform cells
                coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                        device=self.density_grid[t].device)
                indices1 = vren.morton3D(coords1).long()
                # occupied cells
                indices2 = torch.nonzero(self.density_grid[t, c] > density_threshold)[:, 0]
                if len(indices2) > 0:
                    rand_idx = torch.randint(len(indices2), (M,),
                                             device=self.density_grid[t].device)
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
            t_cells.append(cells)
        return t_cells

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
        self.count_grid = torch.zeros_like(self.density_grid[0])
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
                self.density_grid[:, c, indices[i:i + chunk]] = \
                    torch.where(valid_mask, 0., -1.)  # same for all time stamps

    # TODO: consider move updating density grid to forward region to perform AMP

    '''
    updating desntity_grid_should be in float32 operation, this may lead to performance issue.
    consider warp this operation in the forward function, to let lightning module call mixed precision plugin.
    may bring some performance bonus.
    '''

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        # print(f'updating density grid, warmup={warmup}')

        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            t_cells = self.sample_uniform_and_occupied_cells(self.grid_size ** 3 // 4,
                                                             density_threshold)
        for t_ in range(self.time_grid_resolution):
            if not warmup:
                cells = t_cells[t_]
            density_grid_tmp = torch.zeros_like(self.density_grid[t_])

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
                '''
                according to the blendring together algorithm in Neural Scene Flow Fields
                we should add the static and dynamic density together.
                '''
                '''
                generate uniform random t in the range
                '''
                t_interval = self.t_max - self.t_min
                t_start_ = t_interval * (t_ / self.time_grid_resolution) + self.t_min
                t_end = t_interval * ((t_ + 1) / self.time_grid_resolution) + self.t_min
                rand_t = torch.rand_like(xyzs_w[:, 0:1]) * (t_end - t_start_) + t_end

                if self.static_only:
                    rand_t = torch.zeros_like(rand_t)
                density_grid_tmp[c, indices] = self.dynamic_density(xyzs_w, rand_t, return_feat=False).float()
            # TODO: consider the decay.
            '''
            important!!!
            the 0.95 factor of decay may be not suitable for dynamic setting.
            0.95 is very good for static setting.
            '''
            if erode:
                # My own logic. decay more the cells that are visible to few cameras
                decay = torch.clamp(decay ** (1 / self.count_grid), 0.1, 0.95)
            self.density_grid[t_] = \
                torch.where(self.density_grid[t_] < 0,
                            self.density_grid[t_],
                            torch.maximum(self.density_grid[t_] * decay, density_grid_tmp))
            # print(2)
            mean_density = self.density_grid[t_][self.density_grid[t_] > 0].mean().item()
            # print(3)
            vren.packbits(self.density_grid[t_], min(mean_density, density_threshold),
                          self.density_bitfield[t_])

    def get_t_grid_indices(self, batched_time_stamps):
        t_min = - self.time_scale  # -1 as default
        t_max = self.time_scale  # 1 as default

        diff = (batched_time_stamps - t_min) / (t_max - t_min)  # compress into [0,1) to avoid coordinates overflow
        diff_indx = torch.floor(diff * self.time_grid_resolution)  # [0, t_resolution)
        diff_indx = diff_indx.int()
        return torch.clamp(diff_indx, 0, self.time_grid_resolution - 1)
