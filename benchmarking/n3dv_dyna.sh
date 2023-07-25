#!/bin/bash


prefix="time_grids_fix____2"

dash="____"
#for dir in "cut_roasted_beef" "flame_steak" "sear_steak" "flame_salmon_1" "cook_spinach"
for dir in     "cut_roasted_beef" "flame_steak" "sear_steak" "flame_salmon_1" "cook_spinach" "coffee_martini"
do

model_type=1
echo "double loss vary"

for loss_w_ in 0.001 0.005 0.01 0.025 0.05 0.1
do
echo "loss_w_"
echo $loss_w_
# tune opacity loss
python train_dynamic.py \
--model_type $model_type \
--root_dir ./data/n3dv/$dir \
--exp_name $prefix$dir$dash$model_type \
--dataset_name n3dv2 \
--distortion_loss_w 0.0 \
--opacity_loss_w $loss_w_ \
--entropy_loss_w $loss_w_ \
--num_epochs 60 \
--batch_size 8192 \
--eval_lpips \
--regenerate 0 \
 2>&1 | tee -a $prefix$dir$dash$model_type.log
done





echo "\n\n # tune entropyloss \n\n"

for entropy_loss_w_ in 0.005 0.05 0.0005 0.00005
do
echo "entropy_loss_w_"
echo $entropy_loss_w_
python train_dynamic.py \
--model_type $model_type \
--root_dir ./data/n3dv/$dir \
--exp_name $prefix$dir$dash$model_type \
--dataset_name n3dv2 \
--distortion_loss_w 0.0 \
--opacity_loss_w 0.001 \
--entropy_loss_w $entropy_loss_w_ \
--num_epochs 60 \
--batch_size 8192 \
--eval_lpips \
--regenerate 0 \
 2>&1 | tee -a $prefix$dir$dash$model_type.log
done

echo "\n\n# tune opacity \n\n"

for opacity_loss_w_ in 0.005 0.05 0.0005 0.00005
do
echo "opacity_loss_w_"
echo $opacity_loss_w_
# tune opacity loss
python train_dynamic.py \
--model_type $model_type \
--root_dir ./data/n3dv/$dir \
--exp_name $prefix$dir$dash$model_type \
--dataset_name n3dv2 \
--distortion_loss_w 0.0 \
--opacity_loss_w $opacity_loss_w_ \
--entropy_loss_w 0.001 \
--num_epochs 60 \
--batch_size 8192 \
--eval_lpips \
--regenerate 0 \
 2>&1 | tee -a $prefix$dir$dash$model_type.log
done





done