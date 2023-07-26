from tbparse import SummaryReader

import glob
import os
import pandas as pd


random_seeds=[0,1,2,3,42,1337]



def arange_one_exp(root_dir):
    print(f'root dir ={root_dir}')
    dirs_ = sorted(glob.glob(os.path.join(root_dir,"./*/run/ngp")))

    psnr_=[]
    ssim_=[]
    ms_ssim_=[]
    lpips_=[]

    exp_names = []

    for dir_ in dirs_:


        print(f'exp name = {dir_}')
        # extract the exp name from dir
        exp_name = dir_.split('/')[-3]
        exp_names.append(exp_name)
        log_dir = dir_
        reader = SummaryReader(log_dir)
        df = reader.scalars

        rows = df[df['tag'].str.startswith('evaluate')]

        # get the unique tags
        tag_names = list(rows['tag'].unique())
        # get the last row, of every unique tag_name
        last_rows = rows.groupby('tag').last()
        # assert the tags are sorted
        assert list(last_rows.index) == tag_names
        print(last_rows)
        # get the values of the last row, for every unique tag_name
        values = last_rows['value'].values
        print(values    )


        # ASSERT SORTED with character ascending order
        lpips,ms_ssim,psnr,ssim = values
        psnr_.append(psnr)
        ssim_.append(ssim)
        ms_ssim_.append(ms_ssim)
        lpips_.append(lpips)


    print(f'exp_names {exp_names}')
    print(f"psnr: {psnr_}") 
    print(f"ssim: {ssim_}")
    print(f"ms-ssim: {ms_ssim_}")
    print(f"lpips: {lpips_}")
    # append the average of the exps
    if len(psnr_) != 8:
        print('error')

    exp_names.append('average')
    psnr_.append(sum(psnr_)/len(psnr_))
    ssim_.append(sum(ssim_)/len(ssim_))
    ms_ssim_.append(sum(ms_ssim_)/len(ms_ssim_))
    lpips_.append(sum(lpips_)/len(lpips_))
    # assert there are 8 exps and 1 average


    # save the data in csv, at the root_dir
    df = pd.DataFrame({'exp_name':exp_names,'psnr':psnr_,'ssim':ssim_,'ms_ssim':ms_ssim_,'lpips':lpips_})
    df.to_csv(os.path.join(root_dir,'results.csv'),index=False)

    # return the average values
    return psnr_[-1],ssim_[-1],ms_ssim_[-1],lpips_[-1]


def run_different_seeds(exp_name):

    # get the average values for every seed
    psnr=[]
    ssim=[]
    ms_ssim=[]
    lpips=[]

    for seed in random_seeds:
        root_dir = f'./{exp_name}{seed}'
        assert os.path.isdir(root_dir)

    for seed in random_seeds:
        
        root_dir = f'./{exp_name}{seed}'
        psnr_,ssim_,ms_ssim_,lpips_ = arange_one_exp(root_dir)
        psnr.append(psnr_)
        ssim.append(ssim_)
        ms_ssim.append(ms_ssim_)
        lpips.append(lpips_)
    
    print(f'psnr: {psnr}')
    print(f'ssim: {ssim}')
    print(f'ms_ssim: {ms_ssim}')
    print(f'lpips: {lpips}')

    #print average values
    print(f'---------------------------exp_results_average-----------------------------------')
    print(exp_name)
    print(f'psnr: {sum(psnr)/len(psnr)}')
    print(f'ssim: {sum(ssim)/len(ssim)}')
    print(f'ms_ssim: {sum(ms_ssim)/len(ms_ssim)}')
    print(f'lpips: {sum(lpips)/len(lpips)}')
    


if __name__ == '__main__':
    exp_names=[
        'trial_ashawkey_model_',
    #    'trial_new_decay_leaky_',
    #    'trial_simple_weight_decay',
    #    'trial_wdcay_tune',
    #    'trial_tune3_wdcay',
        'trial_tune4_wdcay',
        'trial_tune5_wdcay',
        'trial_tune6_wdcay',

    ]

    for exp_name in exp_names:
        for seed in random_seeds:
            root_dir = f'./{exp_name}{seed}'
            assert os.path.isdir(root_dir)
    print('dirs are ok')
    for exp_name in exp_names:

        run_different_seeds(exp_name)