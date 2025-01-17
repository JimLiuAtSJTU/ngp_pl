import torch
from .custom_functions import \
    RayAABBIntersector, RayMarcher, VolumeRenderer,VolumeRenderer_2
from einops import rearrange
import vren

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


BACKGROUND_FIELD=False


from .debug_utils import nan_check,nan_dict_check



def sigma_entropy_function(x:torch.Tensor):
    # 0 -> 0
    # 0.5 -> 0.5
    # 1.34 -> 0.7
    # 3.24 -> 0.4
    # 5 -> 0.155
    # 10 -> 5e-3
    # 15 -> 1.5e04
    # 20 -> 3.6e-6
    y=torch.clip(x,min=0,max=10)/10.0

    #return 0.5-torch.abs(y-0.5)
    return torch.special.entr(y)




@torch.cuda.amp.autocast()
def render(model, rays_o, rays_d, **kwargs):
    """
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    test_time=kwargs.get('test_time', False)
    if test_time:
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train

    trunk_= kwargs.get('trunks')

    batch_size = rays_o.shape[0]


    if not test_time or trunk_ is None:

        rays_o = rays_o.contiguous(); rays_d = rays_d.contiguous()
        _, hits_t, _ = \
            RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
        hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE


        result = render_func(model, rays_o, rays_d, hits_t, **kwargs)
        for k, v in result.items():
            if kwargs.get('to_cpu', False):
                v = v.cpu()
                if kwargs.get('to_numpy', False):
                    v = v.numpy()
            result[k] = v
    else:
        assert isinstance(trunk_,int)
        # test, and trunks is not None
        rays_o = rays_o.contiguous()
        rays_d = rays_d.contiguous()

        len_ = rays_o.shape[0]
        result={}

        for i in range(0,len_,trunk_):

            start_=i
            end_=min(i+trunk_,len_)
            rays_o__=rays_o[start_:end_]
            rays_d__=rays_d[start_:end_]
            #print(i)
            _, hits_t, _ = \
                RayAABBIntersector.apply(rays_o__, rays_d__, model.center, model.half_size, 1)
            hits_t[(hits_t[:, 0, 0] >= 0) & (hits_t[:, 0, 0] < NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

            rst = render_func(model, rays_o__, rays_d__, hits_t, **kwargs)

            for key in rst.keys():
                if result.get(key) is None:
                    result[key] = []

                tmp=rst.get(key)
                if len(tmp.shape)>0:
                    result[key] += [tmp]


        for key in result.keys():
            try:
                result[key] = torch.cat(result[key], dim=0)
            except:
                assert len(result[key])==0


        for k, v in result.items():
            if kwargs.get('to_cpu', False):
                v = v.cpu()
                if kwargs.get('to_numpy', False):
                    v = v.numpy()
            result[k] = v

    return result


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples 
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)
    if BACKGROUND_FIELD:
        kwargs['rays_o']=rays_o
        kwargs['rays_d']=rays_d
        kwargs['background_field']=True

        '''
        fake forward to get environment RGB value!
        '''
        env_RGB=model(0, 0,**kwargs)
        kwargs.pop('rays_o')
        kwargs.pop('rays_d')
        kwargs.pop('background_field')


    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor==0 else 4

    extra={}
    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive==0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays//N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs==0, dim=1)
        if valid_mask.sum()==0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        sigmas[valid_mask], _rgbs , extra_ = model(xyzs[valid_mask], dirs[valid_mask], **kwargs)
        rgbs[valid_mask] = _rgbs.float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)


        #for key in extra_.keys():
        #    if extra.get(key) is None:
        #        extra[key]=extra_[key]
        #    else:
        #        extra[key]= torch.cat([extra[key],extra_[key]],dim=0)


        vren.composite_test_fw(
            sigmas, rgbs, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rgb)
        alive_indices = alive_indices[alive_indices>=0] # remove converged rays


    results['opacity'] = opacity
    results['depth'] = depth
    if BACKGROUND_FIELD:
        T_inf = 1- opacity
        results['rgb'] = rgb + T_inf[:,None]*env_RGB
    else:
        results['rgb'] = rgb
    results['total_samples'] = total_samples # total samples for all rays

    if exp_step_factor==0: # synthetic
        rgb_bg = torch.ones(3, device=device)
    else: # real
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')

    return results


def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
    """
    time_grid_indx = kwargs.get('t_grid_indx',0)
    exp_step_factor = kwargs.get('exp_step_factor', 0.)

    #print(rays_o,rays_d)
    #print(rays_o.shape,rays_d.shape)
    #exit(0)
    nan_check(rays_o)
    nan_check(rays_d)
    nan_check(hits_t)


    # just by rays_o, rays_d and fourier encoding.
    #
    results = {}
    (rays_a, xyzs, dirs,
    results['deltas'], results['ts'], results['rm_samples']) = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield[time_grid_indx],
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)
    nan_dict_check(results)
    nan_check(rays_a)

    if BACKGROUND_FIELD:
        kwargs['rays_o']=rays_o
        kwargs['rays_d']=rays_d
        kwargs['background_field']=True

        '''
        fake forward to get environment RGB value!
        '''
        env_RGB=model(xyzs, dirs,**kwargs)
        kwargs.pop('rays_o')
        kwargs.pop('rays_d')
        kwargs.pop('background_field')


    #print(f'rays_a,rays_a.shape',rays_a,rays_a.shape)
    #print(kwargs)
    for k, v in kwargs.items():
        # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
    #        print(f'key ={k},value={v}')
    #        print(f'tmp v,{v},{v.shape}')
    #        tmp=v[rays_a[:, 0]]

            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    sigmas, rgbs,extra = model(xyzs, dirs, **kwargs)

    nan_check(sigmas)
    nan_check(rgbs)
    nan_dict_check(extra)

    (results['vr_samples'], results['opacity'],
    results['depth'], results['rgb'], results['ws']) = \
        VolumeRenderer.apply(sigmas, rgbs.contiguous(), results['deltas'], results['ts'],
                             rays_a, kwargs.get('T_threshold', 1e-4))
    try:
        (results['vr_samples_dynamic'], results['opacity_dynamic'],
        results['depth_dynamic'], results['rgb_dynamic'], results['ws_dynamic']) = \
            VolumeRenderer_2.apply(extra['sigma_dynamic'], extra['rgb_dynamic'].contiguous(), results['deltas'], results['ts'],
                                 rays_a, kwargs.get('T_threshold', 1e-4))
    except:
        results['vr_samples_dynamic']=results['vr_samples']

        results['opacity'] = results['opacity']
        results['depth'] = results['depth']
        results['rgb'] = results['rgb']
        results['rgb'] = results['rgb']


    '''
    next step may be to optimize the reconstruction quality by using the "far background field" in SUDS.
    may be written in cuda or pure pytorch.
    
    t_inf= torch.exp( - torch.sum(sigma*delta))
    
    
    
    rgb += t_inf* rgb_inf()
    
    '''

    results['rays_a'] = rays_a

    if exp_step_factor==0: # synthetic
        rgb_bg = torch.ones(3, device=rays_o.device)
        results['rgb'] = results['rgb'] + \
                         rgb_bg * rearrange(1 - results['opacity'], 'n -> n 1')

    else: # real
        if kwargs.get('random_bg', False):
            rgb_bg = torch.rand(3, device=rays_o.device)
            results['rgb'] = results['rgb'] + \
                             rgb_bg * rearrange(1 - results['opacity'], 'n -> n 1')

        else:
            rgb_bg = torch.zeros(3, device=rays_o.device)

    if BACKGROUND_FIELD:
        T_inf = 1 - results['opacity']
        results['rgb'] = results['rgb']+ T_inf[:,None] * env_RGB
    results.update(extra)

    results['sigma_entropy'] = sigma_entropy_function(sigmas)

    nan_dict_check(results)
    return results
