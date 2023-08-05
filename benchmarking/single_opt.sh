#!/bin/bash





#python gpu_wait.py



prefix="tune_bsize"

for dir in     "cut_roasted_beef" "flame_salmon_1"  "flame_steak" "sear_steak" "cook_spinach" "coffee_martini"
#for dir in      "flame_salmon_1"   "cook_spinach"

do

b_size=768

python train_dynamic.py --root_dir /home/ubuntu/datasets/zhenhuanliu/ngp_pl/data/n3dv/$dir \
--exp_name \
$prefix$dir$b_size \
--dataset_name \
n3dv2 \
--num_epochs \
600 \
--regenerate \
0 \
--update_interval 16 \
--model_type \
1 \
--batch_size \
$b_size \
2>&1 | tee -a $prefix$dir$b_size.log




done


