import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
import tinycudann as tcnn
import numpy as np

n_slope=0.1

class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_deform="frequency",  # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_deform=5,  # a deeper MLP is very necessary for performance.
                 hidden_dim_deform=128,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # deformation network
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform
        self.encoder_deform, self.in_dim_deform = get_encoder(encoding_deform, multires=10)
        self.encoder_time, self.in_dim_time = get_encoder(encoding_time, input_dim=1, multires=6)

        deform_net = []
        for l in range(num_layers_deform):
            if l == 0:
                in_dim = self.in_dim_deform + self.in_dim_time  # grid dim + time
            else:
                in_dim = hidden_dim_deform

            if l == num_layers_deform - 1:
                out_dim = 3  # deformation for xyz
            else:
                out_dim = hidden_dim_deform

            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.in_dim_time + self.in_dim_deform  # concat everything
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19,
                                                          desired_resolution=2048)  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]

        # deform
        enc_ori_x = self.encoder_deform(x, bound=self.bound)  # [N, C]
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1)  # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1)  # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)

        x = x + deform

        # sigma
        x = self.encoder(x, bound=self.bound)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return sigma, rgbs, deform

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        results = {}

        # deformation
        enc_ori_x = self.encoder_deform(x, bound=self.bound)  # [N, C]
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1)  # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1)  # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.relu(deform, inplace=True)

        x = x + deform
        results['deform'] = deform

        # sigma
        x = self.encoder(x, bound=self.bound)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat

        return results

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

        # optimizer utils

    def get_params(self, lr, lr_net):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_deform.parameters(), 'lr': lr},
            {'params': self.encoder_time.parameters(), 'lr': lr},
            {'params': self.deform_net.parameters(), 'lr': lr_net},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})

        return params


class NeRFNetwork_2(NeRFRenderer):
    def __init__(self,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_deform="frequency",  # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=47,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_deform=5,  # a deeper MLP is very necessary for performance.
                 hidden_dim_deform=128,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # deformation network
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform
        self.encoder_deform, self.in_dim_deform = get_encoder(encoding_deform, multires=10)
        self.encoder_time, self.in_dim_time = get_encoder(encoding_time, input_dim=1, multires=6)
        # self.in_dim_time = 13
        deform_net = []
        for l in range(num_layers_deform):
            if l == 0:
                in_dim = self.in_dim_deform + self.in_dim_time  # grid dim + time
            else:
                in_dim = hidden_dim_deform

            if l == num_layers_deform - 1:
                out_dim = 3  # deformation for xyz
            else:
                out_dim = hidden_dim_deform

            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        self.grid_encoder = grid_encoder_4d()

        L = 2;
        F = 16;
        log2_T = 7;
        N_min = 30
        highest_reso = 100  # lower than the dimension
        b = np.exp(np.log(highest_reso / N_min) / (L - 1))
        self.latent_code_feature_dim=L*F

        #b=1
        self.time_latent_code = tcnn.Encoding(
            n_input_dims=1,
            encoding_config={
                        "otype": "Grid",
                        "type": "Tiled",
                        "n_levels": L,
                        "n_features_per_level": F,
                        "log2_hashmap_size": log2_T,
                        "base_resolution": N_min,
                        "per_level_scale": b,
                        "interpolation": "Linear"
            }
        )


        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.in_dim_deform + self.in_dim_time  # concat everything
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim + self.latent_code_feature_dim+self.in_dim_time
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19,
                                                          desired_resolution=2048)  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]

        # deform
        enc_ori_x = self.encoder_deform(x, bound=self.bound)  # [N, C]

        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1)  # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1)  # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.leaky_relu(deform, inplace=True,negative_slope=n_slope)

        x = x + deform

        # sigma
        x = self.encoder(x, bound=self.bound)
        t_ = t.expand(x.shape[0], 1)

        # x = self.grid_encoder(x,t_)
        t_code = self.time_latent_code(t_)
        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.leaky_relu(h, inplace=True,negative_slope=n_slope)

        #sigma = F.relu(h[..., 0])
        sigma = F.relu(trunc_exp(h[..., 0])-0.5)
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat,t_code,enc_t], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.leaky_relu(h, inplace=True,negative_slope=n_slope)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return sigma, rgbs, deform

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        results = {}

        # deformation
        enc_ori_x = self.encoder_deform(x, bound=self.bound)  # [N, C]
        enc_t = self.encoder_time(t)  # [1, 1] --> [1, C']
        if enc_t.shape[0] == 1:
            enc_t = enc_t.repeat(x.shape[0], 1)  # [1, C'] --> [N, C']

        deform = torch.cat([enc_ori_x, enc_t], dim=1)  # [N, C + C']
        for l in range(self.num_layers_deform):
            deform = self.deform_net[l](deform)
            if l != self.num_layers_deform - 1:
                deform = F.leaky_relu(deform, inplace=True,negative_slope=n_slope)

        x = x + deform
        results['deform'] = deform

        # sigma
        x = self.encoder(x, bound=self.bound)
        t_=t.expand(x.shape[0],1)

        # x = self.grid_encoder(x,t_)

        t_code=self.time_latent_code(t_)
        # t_code and enc_t
        # seems like an attention



        h = torch.cat([x, enc_ori_x, enc_t], dim=1)
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.leaky_relu(h, inplace=True,negative_slope=n_slope)

        #sigma = F.relu(h[..., 0])
        sigma = F.relu(trunc_exp(h[..., 0])-0.5)
        geo_feat = h[..., 1:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat

        return results

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x)  # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

        # optimizer utils

    def get_params(self, lr, lr_net):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr_net,
             'weight_decay':1e-9,
             'betas': (0.9, 0.999)

             },
            {'params': self.encoder_deform.parameters(), 'lr': lr},
            {'params': self.encoder_time.parameters(), 'lr': lr},
            {'params': self.time_latent_code.parameters(), 'lr': lr_net,
             'weight_decay': 1e-9,
             'betas': (0.9, 0.999)

             },
            {'params': self.deform_net.parameters(), 'lr': lr_net},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})

        return params



class NeRFNetwork_3(NeRFNetwork_2):
    def __init__(self,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_deform="frequency",  # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=47,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_deform=5,  # a deeper MLP is very necessary for performance.
                 hidden_dim_deform=128,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(encoding,encoding_dir,encoding_time,encoding_deform,encoding_bg,num_layers,hidden_dim,geo_feat_dim,num_layers_color,hidden_dim_color,num_layers_bg,
                         hidden_dim_bg,num_layers_deform,hidden_dim_deform,bound,
                         **kwargs)
    def get_params(self, lr, lr_net):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr_net},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr_net,
             'weight_decay':5e-3,
             'betas':(0.9,0.999)
             },
            {'params': self.encoder_deform.parameters(), 'lr': lr},
            {'params': self.encoder_time.parameters(), 'lr': lr},
            {'params': self.time_latent_code.parameters(), 'lr': lr_net,
             'weight_decay':5e-3,
             'betas':(0.9,0.999)
             },
            {'params': self.deform_net.parameters(), 'lr': lr_net},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})

        return params




class grid_encoder_4d(torch.nn.Module):
    def __init__(self):
        super().__init__()

        L = 2;
        F = 20;  # 40 dim
        log2_T = 9;  # 256 hash tables.
        N_min = 120  # 300 frames, each part = 50framse   total, 10s.
        highest_reso = 150 * 0.666  # lower than the dimension
        b = np.exp(np.log(highest_reso / N_min) / (L - 1))

        self.time_latent_code = tcnn.NetworkWithInputEncoding(
            n_input_dims=1, n_output_dims=40,
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

        L = 12;
        F = 2;
        log2_T = 19;
        N_min = 16
        b = np.exp(np.log(2048 * 1 / N_min) / (L - 1))

        self.grid_4d_encoder = tcnn.NetworkWithInputEncoding(
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

        self.xyz_encoding = tcnn.Encoding(
            n_input_dims=3,
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

        self.fusion_mlp = tcnn.Network(
            n_input_dims=32 + 40, n_output_dims=32,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )
        self.encoder, self.in_dim = get_encoder('tiledgrid', desired_resolution=2048 * 1)

    def forward(self, x, t):
        x_code = self.encoder(x)
        t_code = self.time_latent_code(t)
        t_code = torch.zeros_like(t_code)
        concat_ = torch.cat([x_code, t_code], -1)
        tmp = self.fusion_mlp(concat_)
        return tmp
        # return self.grid_4d_encoder(torch.cat([x,t[:,2:]],-1))


class NeRFNetwork2(NeRFRenderer):
    def __init__(self,
                 encoding="tiledgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_time="frequency",
                 encoding_deform="frequency",  # "hashgrid" seems worse
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 num_layers_deform=3,  # a deeper MLP is very necessary for performance.
                 hidden_dim_deform=128,
                 bound=1,
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # deformation network
        self.num_layers_deform = num_layers_deform
        self.hidden_dim_deform = hidden_dim_deform
        self.encoder_deform, self.in_dim_deform = get_encoder(encoding_deform, multires=10)
        self.encoder_time, self.in_dim_time = get_encoder(encoding_time, input_dim=1, multires=6)

        deform_net = []
        for l in range(num_layers_deform):
            if l == 0:
                in_dim = self.in_dim_deform + self.in_dim_time  # grid dim + time
            else:
                in_dim = hidden_dim_deform

            if l == num_layers_deform - 1:
                out_dim = 3  # deformation for xyz
            else:
                out_dim = hidden_dim_deform

            deform_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.deform_net = nn.ModuleList(deform_net)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.encoder, self.in_dim = get_encoder(encoding, desired_resolution=2048 * bound)
        L = 16;
        F = 2;
        log2_T = 19;
        N_min = 16
        b = np.exp(np.log(2048 * self.bound / N_min) / (L - 1))

        self.grid_encoder_4d = grid_encoder_4d()

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim + self.in_dim_time + self.in_dim_deform  # concat everything
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = 48 + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim_color

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19,
                                                          desired_resolution=2048)  # much smaller hashgrid

            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg

                if l == num_layers_bg - 1:
                    out_dim = 3  # 3 rgb
                else:
                    out_dim = hidden_dim_bg

                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    def forward(self, x, d, t):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]
        # t: [1, 1], in [0, 1]

        # deform
        # t_ = t.expand(x.shape[0], 1)

        t_ = x.clone()
        t_[:, 2] = t

        results = {}

        # deformation

        deform = torch.FloatTensor([0])

        # sigma
        h = self.grid_encoder_4d(x, t_)

        # sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return sigma, rgbs, deform

    def density(self, x, t):
        # x: [N, 3], in [-bound, bound]
        # t: [1, 1], in [0, 1]

        t_ = x.clone()
        t_[:, 2] = t

        results = {}

        # deformation

        results['deform'] = torch.FloatTensor([0])

        # sigma
        h = self.grid_encoder_4d(x, t_)

        sigma = F.relu(h[..., 0])
        #sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        results['sigma'] = sigma
        results['geo_feat'] = geo_feat

        return results

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # t: [1, 1], in [0, 1]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.
        raise NotImplementedError
        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device)  # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype)  # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

        # optimizer utils

    def get_params(self, lr, lr_net):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr_net},
            {'params': self.grid_encoder_4d.parameters(), 'lr': lr},
            {'params': self.encoder_time.parameters(), 'lr': lr},
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr_net})

        return params
