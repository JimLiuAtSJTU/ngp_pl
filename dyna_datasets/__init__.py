from .nerf import NeRFDataset
from .nsvf import NSVFDataset
from .colmap import ColmapDataset
from .nerfpp import NeRFPPDataset
from .n3dv_llff import N3DV_dataset,N3DV_dataset_2
from .rtmv import RTMVDataset


dataset_dict = {'nerf': NeRFDataset,
                'nsvf': NSVFDataset,
                'colmap': ColmapDataset,
                'nerfpp': NeRFPPDataset,
                'n3dv': N3DV_dataset,
                'n3dv2': N3DV_dataset_2,

                'rtmv': RTMVDataset}