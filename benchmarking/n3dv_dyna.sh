#!/bin/bash


prefix="dynamic4"

#for dir in "cut_roasted_beef" "flame_steak" "sear_steak" "flame_salmon_1" "cook_spinach"
for dir in   "coffee_martini"  "cut_roasted_beef" "flame_steak" "sear_steak" "flame_salmon_1" "cook_spinach"

do
python train_dynamic.py \
--root_dir ./data/n3dv/$dir \
--exp_name $dir \
--dataset_name n3dv2 \
--distortion_loss_w 0.000 \
--num_epochs 15 \
--regenerate 1 \
 2>&1 | tee -a $prefix$dir.log



done