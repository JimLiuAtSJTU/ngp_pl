#!/bin/bash









prefix="importance_sampling_"

for dir in     "cut_roasted_beef" "flame_salmon_1"  "flame_steak" "sear_steak" "cook_spinach" "coffee_martini"
#for dir in      "flame_salmon_1"   "cook_spinach"

do

python train_dynamic.py --root_dir ./data/n3dv/$dir \
--exp_name \
$prefix$dir \
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
512 \
2>&1 | tee -a $prefix$dir.log

done