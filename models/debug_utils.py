
import torch




check_=True

def nan_check(x):
    if not check_:return
    if isinstance(x,torch.Tensor):
        try:
            assert not torch.any(torch.isnan(x))
            assert not torch.any(torch.isinf(x))
        except:

            print('nan occured! ')
            print(torch.sum(torch.isnan(x))/torch.sum(torch.ones_like(x)))
            raise AssertionError

    else:
        pass


def nan_dict_check(x:dict):
    if not check_:return

    for k,v in x.items():
        try:
            nan_check(v)
        except:
            print('nan occured! ')
            print(torch.sum(torch.isnan(v))/torch.sum(torch.ones_like(v)))
            print(k,v)

            raise AssertionError


