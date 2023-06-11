#!/bin/bash









prefix="avoid_nan__"

for dir in     "cut_roasted_beef" "flame_salmon_1"  "flame_steak" "sear_steak" "cook_spinach" "coffee_martini"
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
--model_type \
1 \
--batch_size \
512 \
2>&1 | tee -a $prefix$dir.log



done