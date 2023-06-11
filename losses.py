import torch
from torch import nn
import vren


class DistortionLoss(torch.autograd.Function):
    """
    Distortion loss proposed in Mip-NeRF 360 (https://arxiv.org/pdf/2111.12077.pdf)
    Implementation is based on DVGO-v2 (https://arxiv.org/pdf/2206.05085.pdf)

    Inputs:
        ws: (N) sample point weights
        deltas: (N) considered as intervals
        ts: (N) considered as midpoints
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]

    Outputs:
        loss: (N_rays)
    """
    @staticmethod
    def forward(ctx, ws, deltas, ts, rays_a):
        loss, ws_inclusive_scan, wts_inclusive_scan = \
            vren.distortion_loss_fw(ws, deltas, ts, rays_a)
        ctx.save_for_backward(ws_inclusive_scan, wts_inclusive_scan,
                              ws, deltas, ts, rays_a)
        return loss

    @staticmethod
    def backward(ctx, dL_dloss):
        (ws_inclusive_scan, wts_inclusive_scan,
        ws, deltas, ts, rays_a) = ctx.saved_tensors
        dL_dws = vren.distortion_loss_bw(dL_dloss, ws_inclusive_scan,
                                         wts_inclusive_scan,
                                         ws, deltas, ts, rays_a)
        return dL_dws, None, None, None


#Element_Entropy=torch.special.entr


def Element_Entropy(x:torch.Tensor):
    y=torch.clamp(x,min=1e-7,max=1) # clamp to avoid nan
    return -y*torch.log(y)



class NeRFLoss(nn.Module):
    def __init__(self, lambda_opacity=1e-3, lambda_distortion=1e-3,lambda_entropy= 0.001,sigma_entropy=1e-7,lambda_opac_dyna=1e-7):
        super().__init__()

        self.lambda_opacity = lambda_opacity
        self.lambda_distortion = lambda_distortion
        self.lambda_entropy = lambda_entropy
        self.lambda_sigma_entropy = sigma_entropy
        self.lambda_opac_dyna=lambda_opac_dyna
    def forward(self, results, target,use_dst_loss=False, **kwargs):
        d = {}
        batch_size=results['rgb'].shape[0]

        start_=target.get('start',0)
        end_=target.get('end',batch_size)
        d['rgb'] = torch.mean((results['rgb']-target['rgb'][start_:end_])**2)

        static_weight=results['static_weight']
        #print(f'weight{static_weight}')
        '''
        do not use symmetric binary entropy
        because we encourage static_weight -> 1
        '''
        #print(f'static weight={static_weight},{static_weight.shape}')
        entropyloss= torch.mean(Element_Entropy(static_weight)) #  Element_Entropy(1-static_weight)

        o = results['opacity']+1e-10

        opacity_dynamic=results['opacity_dynamic'] + 1e-10

        sigma_entropy = results['sigma_entropy']
        # encourage opacity to be either 0 or 1 to avoid floater
        d['opacity'] = torch.mean((-o*torch.log(o)))*self.lambda_opacity
        d['opacity_dynamic'] =  torch.mean((-opacity_dynamic*torch.log(opacity_dynamic)))*self.lambda_opac_dyna/1000

        d['sigma_entropy'] = torch.mean((sigma_entropy))*self.lambda_sigma_entropy

        if self.lambda_distortion > 0 and use_dst_loss:
            d['distortion'] = self.lambda_distortion * \
                DistortionLoss.apply(results['ws'], results['deltas'],
                                     results['ts'], results['rays_a'])
        d['entropy']=entropyloss*self.lambda_entropy
        #d['dynamic']=  torch.sum(torch.abs(1-static_weight))*self.lambda_entropy


        return d

'''
sum the loss
and divide by num_of_t_trunks=batch_size//t_trunk_size
to avoid concat
'''
def loss_sum(loss_A:(dict,None),loss_B:dict):
    if loss_A is None:
        return loss_B
    assert len(loss_A.keys())==len(loss_B.keys())
    result=loss_A

    for k in loss_A.keys():
        result[k]+=loss_B[k]

    #print(f'result={result},lossA={loss_A},lossb={loss_B}')
    return result

def dict_sum(loss_A:(dict,None),loss_B:dict,keys=None):
    if loss_A is None or len(loss_A)==0:
        return loss_B
    if keys is None:
        assert len(loss_A.keys())==len(loss_B.keys())
    result={}

    if keys is not None:
        keys_set=keys
    else:
        keys_set=loss_A.keys()
    for k in keys_set:
        result[k]=loss_A[k]+loss_B[k]
    return result

