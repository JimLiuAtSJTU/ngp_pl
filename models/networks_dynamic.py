import torch
from torch import nn
import tinycudann as tcnn
import vren
from einops import rearrange
from .custom_functions import TruncExp
import numpy as np

from .rendering import NEAR_DISTANCE


class NGP_4D(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid'):
        super().__init__()

        self.rgb_act = rgb_act

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
                             torch.zeros(self.cascades * self.grid_size ** 3 // 8, dtype=torch.uint8))

        # constants
        L = 18;
        F = 2;
        log2_T = 19;
        N_min = 16
        b = np.exp(np.log(2048 * scale / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=4, n_output_dims=16,
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
                    "n_hidden_layers": 2,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

        if self.rgb_act == 'None':  # rgb_net output is log-radiance
            for i in range(3):  # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def density(self, x, t=None, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)

        if t is None:
            '''
            t should be [-1,1]
            '''
            #t=torch.zeros_like(x[:,0:1]) # zeros get very low performance
            t  =  -  torch.ones_like(x[:,0:1])


        elif max(t.shape)==1:
            # is scalar
            t=torch.ones_like(x[:,0:1]) * t
        x=torch.cat([x,t],dim=-1)

        h = self.xyz_encoder(x)
        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)


        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else:  # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i + 1] + log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs

    def forward(self, x, d, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """

        # print(f'')

        t=kwargs.get('times')

        sigmas, h = self.density(x,t, return_feat=True)
        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        if self.rgb_act == 'None':  # rgbs is log-radiance
            if kwargs.get('output_radiance', False):  # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else:  # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)
        extra={
        }
        return sigmas, rgbs,extra

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
        # print(f'updating density grid, warmup={warmup}')
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size ** 3 // 4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            try:
                indices, coords = cells[c]
            except TypeError:
                print(cells)
            if indices is None and coords is None:
                continue
            s = min(2 ** (c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords / (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)
        cells.clear()

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


class NGP_time(nn.Module):
    def __init__(self, scale, rgb_act='Sigmoid'):
        super().__init__()

        self.rgb_act = rgb_act
        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
                             torch.zeros(self.cascades * self.grid_size ** 3 // 8, dtype=torch.uint8))

        self.encoder_static=self.__get_hash_encoder(input_dims=3)
        self.encoder_dynamic=self.__get_hash_encoder(input_dims=4)


        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )


        flow_dim=0
        self.rgb_net_static = self.__get_fused_mlp(input_dims=32,output_dims=3)
        self.rgb_net_dynamic = self.__get_fused_mlp(input_dims=32,output_dims=3 + 1 + flow_dim) # rho for another dim


        if self.rgb_act == 'None':  # rgb_net output is log-radiance
            raise NotImplementedError
        print(f'time aware NGP model initialized')

    def __get_hash_encoder(self, input_dims=3):
        # constants
        L = 16;
        F = 2;
        log2_T = 19;
        N_min = 16
        b = np.exp(np.log(2048 * self.scale / N_min) / (L - 1))
        #print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        return  tcnn.NetworkWithInputEncoding(
                n_input_dims=input_dims, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear" # Linear
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

    def __get_fused_mlp(self,input_dims=32,output_dims=3):

        return   tcnn.Network(
            n_input_dims=input_dims, n_output_dims=output_dims,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": self.rgb_act,
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )

    def static_branch(self,x):
        pass
        sigmas=0
        rgbs=0
        return sigmas,rgbs
    def dynamic_branch(self,x,t):
        pass
        sigmas=0
        rgbs=0
        return sigmas,rgbs

    def blend_together(self,s_sigma,d_sigma,s_rgb,d_rgb,rho):
        '''

        SUDS blending.
        #https://arxiv.org/abs/2303.14536
        s_sigma,d_sigma: static, dynamic
        s_rgb,d_rgb: static, dynamic
        rho: shadow factor in [0,1]. consider using a sigmoid.
        '''
        sigma=s_sigma+d_sigma


        eps=1e-6
        w_static=s_sigma/torch.clamp(sigma,min=eps)
        #print(f's_sigma{s_sigma}{s_sigma.shape}')
        #print(f'w_static{w_static}{w_static.shape}')
        #print(f's_rgb{s_rgb}{s_rgb.shape}')

        #print(f'rho{rho,}{rho.shape}')


        # unsqueeze
        rgb=(w_static*(1-rho))[:,None]*s_rgb
        #print(f'rgb,{rgb.shape}')
        #print(f's_rgb,{s_rgb.shape}')

        rgb = rgb +(1-w_static)[:,None]*d_rgb


        #return sigma,rgb
        '''
        
        return static for debug!
        '''
        return sigma,rgb , w_static











    def __xyzt_network(self, x_coords, t):

        yzt=torch.cat([x_coords[:,1:3],t],dim=-1)
        xyt=torch.cat([x_coords[:,0:2],t],dim=-1)
        zxt=torch.cat([x_coords[:,-1],x_coords[:,0],t],dim=-1)

        pass

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

    def dynamic_density(self, x, t=0, return_feat=False):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        x= torch.cat([x,t],dim=-1)
        h = self.encoder_dynamic(x)


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

        t=kwargs.get('times')
        try:
            assert t.shape[0] == x.shape[0]
        except AssertionError:
            #print(f'x,t',x,t)
            #print(f'x,t',x.shape,t.shape)

            assert t.shape[0] == 1

            t=t.expand(x.shape[0],1)
        #except IndexError:
        #    assert isinstance(t,torch.Tensor)
        #    print(f't,{t},{type(t)}')

        d = d / torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d + 1) / 2)


        sigma_static,h_static=self.static_density(x,return_feat=True)
        rgb_static=self.rgb_net_static(torch.cat([d, h_static],1))

        sigma_dynamic,h_dyna=self.dynamic_density(x,t,return_feat=True)
        rgb_dynamic=self.rgb_net_dynamic(torch.cat([d, h_dyna],1))

        extra={}
        sigma,rgb,weight= self.blend_together(s_sigma=sigma_static,
                                   d_sigma=sigma_dynamic,
                                   s_rgb=rgb_static,
                                   d_rgb=rgb_dynamic[:,:-1],
                                   rho=rgb_dynamic[:,-1])
        extra['static_weight']=weight

        return sigma,rgb,extra






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
                indices2=None
                coords2=None

            # concatenate
            #cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]
            #torch.cuda.empty_cache()
            #print(f'coords{coords1},{coords1.shape},{coords2},{coords2.shape}')
            #print(f'indices{indices1},{indices1.shape},{indices2},{indices2.shape}')

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
