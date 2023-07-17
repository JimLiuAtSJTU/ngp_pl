import json
import warnings

import numpy as np
import pdb
import torch

BOX_OFFSETS = torch.tensor([[[i, j, k] for i in [0, 1] for j in [0, 1] for k in [0, 1]]],
                           device='cuda')


def hash(coords, log2_hashmap_size):
    '''
    coords: this function can process upto 7 dim coordinates
    log2T:  logarithm of T w.r.t 2
    '''
    primes = [1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737]

    xor_result = torch.zeros_like(coords)[..., 0]
    for i in range(coords.shape[-1]):
        xor_result ^= coords[..., i] * primes[i]

    return torch.tensor((1 << log2_hashmap_size) - 1).to(xor_result.device) & xor_result



box_min__, box_max__ = torch.tensor([0],device='cuda'),torch.tensor([1],device='cuda')
grid_block__=torch.tensor([1.0, 1.0, 1.0],device='cuda')

def get_voxel_vertices(xyz,  resolution, log2_hashmap_size):
    '''
    xyz: 3D coordinates of samples. B x 3
    bounding_box: min and max x,y,z coordinates of object bbox
    resolution: number of voxels per axis
    '''
    box_min, box_max =box_min__.to(xyz.device), box_max__.to(xyz.device)
    #if not torch.all(xyz <= box_max) or not torch.all(xyz >= box_min):
    #    warnings.warn("ALERT: some points are outside bounding box. Clipping them!")
    #    xyz = torch.clamp(xyz, min=box_min, max=box_max)

    grid_size = (box_max - box_min) / resolution

    bottom_left_idx = torch.floor((xyz - box_min) / grid_size).int()
    voxel_min_vertex = bottom_left_idx * grid_size + box_min

    voxel_max_vertex = voxel_min_vertex + grid_block__ * grid_size

    voxel_indices = bottom_left_idx.unsqueeze(1) + BOX_OFFSETS
    hashed_voxel_indices = hash(voxel_indices, log2_hashmap_size)

    return voxel_min_vertex, voxel_max_vertex, hashed_voxel_indices


