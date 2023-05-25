#!/bin/bash


prefix="cmp_exp_8"

dash="____"
#for dir in "cut_roasted_beef" "flame_steak" "sear_steak" "flame_salmon_1" "cook_spinach"
for dir in   "coffee_martini"  "cut_roasted_beef" "flame_steak" "sear_steak" "flame_salmon_1" "cook_spinach"
do

for model_type in 1 0

do
python train_dynamic.py \
--model_type $model_type \
--root_dir ./data/n3dv/$dir \
--exp_name $prefix$dir$dash$model_type \
--dataset_name n3dv2 \
--distortion_loss_w 0.000 \
--num_epochs 15 \
--regenerate 1 \
 2>&1 | tee -a $prefix$dir$dash$model_type.log
done


done