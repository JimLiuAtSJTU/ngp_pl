#!/bin/bash





python gpu_wait.py




for dir in     "cut_roasted_beef" "flame_salmon_1"  "flame_steak" "sear_steak" "cook_spinach" "coffee_martini"
#for dir in      "flame_salmon_1"   "cook_spinach"

do

b_size=512
#python train_dynamic.py --root_dir /home/ubuntu/datasets/zhenhuanliu/ngp_pl/data/n3dv/$dir \
prefix="comparision_naive_4d_ngp___"

python train_dynamic.py --root_dir ./data/n3dv/$dir \
--exp_name \
$prefix$dir$b_size \
--dataset_name \
n3dv2 \
--num_epochs \
300 \
--regenerate \
0 \
--update_interval 16 \
--model_type \
-1 \
--batch_size \
$b_size \
2>&1 | tee -a $prefix$dir.log


done

