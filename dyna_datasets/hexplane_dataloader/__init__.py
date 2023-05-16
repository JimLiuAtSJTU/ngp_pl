import time

from .dnerf_dataset import DNerfDataset
from .neural_3D_dataset_NDC import Neural3D_NDC_Dataset


def get_train_dataset(cfgs_custom:dict,is_stack=False):
    t0=time.time()

    cfgs={
        'root_dir':'/',
        'downsample':4,
        'time_scale':1,
        'scene_bbox_min':[-2,-2,-2],
        'scene_bbox_max': [2,2,2],
        'nv3d_ndc_bd_factor':1,
        'nv3d_ndc_eval_step':1,
        'nv3d_ndc_eval_index':0
    }

    cfgs.update(cfgs_custom)

    '''
    get boundbox, bd factor, eval_step
    integrate them!
    '''




    train_dataset = Neural3D_NDC_Dataset(
        cfgs['root_dir'],
        "train",
        cfgs['downsample'],
        is_stack=is_stack,
        N_vis=-1,
        time_scale=cfgs['time_scale'],
        scene_bbox_min=cfgs['scene_bbox_min'],
        scene_bbox_max=cfgs['scene_bbox_max'],
        bd_factor=cfgs['nv3d_ndc_bd_factor'],
        eval_step=cfgs['nv3d_ndc_eval_step'],
        eval_index=cfgs['nv3d_ndc_eval_index'],
    )
    t1=time.time()

    print(f'time elapse seconds:{t1-t0}')

    useful_data={
        'rgb':train_dataset.all_rgbs,
        'poses':train_dataset.poses, # N_CAM,
        #'directions':train_dataset.directions,
        'times':train_dataset.all_times, # N_CAM, time_frames, 1
        'K': train_dataset.K,
        'rays':train_dataset.all_rays,
        'img_wh':train_dataset.img_wh,
        'importance': train_dataset.all_importances
    }

    print(f'train dataset')
    return useful_data


def get_test_dataset(cfgs_custom:dict, is_stack=True):

    cfgs={
        'root_dir':'/',
        'downsample':4,
        'time_scale':1,
        'scene_bbox_min':[-2,-2,-2],
        'scene_bbox_max': [2,2,2],
        'nv3d_ndc_bd_factor':1,
        'nv3d_ndc_eval_step':1,
        'nv3d_ndc_eval_index':0
    }

    cfgs.update(cfgs_custom)


    test_dataset = Neural3D_NDC_Dataset(
        cfgs['root_dir'],
        "test",
        cfgs['downsample'],
        is_stack=is_stack,
        N_vis=-1,
        time_scale=cfgs['time_scale'],
        scene_bbox_min=cfgs['scene_bbox_min'],
        scene_bbox_max=cfgs['scene_bbox_max'],
        bd_factor=cfgs['nv3d_ndc_bd_factor'],
        eval_step=cfgs['nv3d_ndc_eval_step'],
        eval_index=cfgs['nv3d_ndc_eval_index'],
    )

    print(f'test dataset')
    useful_data = {
        #'rgb': test_dataset.all_rgbs,
        'poses': test_dataset.poses, # N_CAM, 3,4
        #'directions': test_dataset.directions,
        'times': test_dataset.all_times, # N_CAM, time_frames, 1
        'rays': test_dataset.all_rays,
        'img_wh': test_dataset.img_wh,

        'K': test_dataset.K,
    }

    if len(test_dataset.all_rgbs)>0:
        useful_data['rgb'] =test_dataset.all_rgbs


    return useful_data
