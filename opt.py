import argparse



# remain identical to hexplane
default_downsample=1/(2704/1024)/2
#default_downsample=1/4

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_type', type=int, default=1,choices=[0,1,-1],
                        help='0 for ngp_time, 1 for ngp_time_plus')
    parser.add_argument('--update_interval', type=int, default=16,
                        help='update interval of the grids')


    # dataset parameters
    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv','n3dv','n3dv2','dnerf'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=default_downsample,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=15,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=5e-4,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')
    parser.add_argument('--seed', type=int, default=1337, #8192
                        help='random seed')
    parser.add_argument('--regenerate', type=int, default=0,choices=[0,1] ,
                        help='whether regenerate dataset, 0 for false, 1 for true')

    parser.add_argument('--static_only', type=int, default=0,choices=[0,1] ,
                        help='whether use static data for debug scenario')

    parser.add_argument('--cache_importance_epochs', type=int, default=10 ,
                        help='set the cache size of importance sampled indices. '
                             'default is cache the array with size'
                             ' corresponding to  10 epochs.')

    # training options
    parser.add_argument('--batch_size', type=int, default=512, # 512 is OK
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='hirachy',
                        choices=['all_images', 'same_image','all_time','batch_time','same_time','importance_time_batch','hirachy'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30, # 300
                        help='number of training epochs')
    parser.add_argument('--erode', type=int, default=1,
                        choices=[0,1],
                        help='whether perform complex erode in network')
    parser.add_argument('--sigma_entropy_loss_w', type=float, default=1e-4,
                        help='''weight of sigma_entropy loss (see losses.py), default is 1e-5 to see the entropy
                        ''')

    parser.add_argument('--opacity_loss_w', type=float, default=5e-3,
                        help='''weight of opacity loss (see losses.py), default is 1e-3 as kwea123.
                        ''')
    parser.add_argument('--opacity_loss_dynamic_w', type=float, default=5e-7,
                        help='''weight of dynamic opacity loss (see losses.py), default is 1e-3 as kwea123.
                        ''')

    parser.add_argument('--entropy_loss_w', type=float, default=5e-3,
                        help='''weight of entropy loss (see losses.py), default is 1e-3 tuned like in the static setting.
                        ''')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-3, # 1e-3 is default
                        help='learning rate')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=True,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    return parser.parse_args()
