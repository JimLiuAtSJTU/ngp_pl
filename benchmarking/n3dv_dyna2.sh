#!/bin/bash










# a good choice is distortion loss =0.001,  entropy = opacity =0.005
# enable erode


prefix="time_grids_finetune"

dash="____"
#for dir in "cut_roasted_beef" "flame_steak" "sear_steak" "flame_salmon_1" "cook_spinach"
for dir in     "cut_roasted_beef" "flame_salmon_1"  "flame_steak" "sear_steak" "cook_spinach" "coffee_martini"
do
model_type=1
echo "double loss vary"

for loss_w_ in 0.0005 0.001 0.005 0.01 0.025 0.05
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

python train_dynamic.py \
--model_type $model_type \
--root_dir ./data/n3dv/$dir \
--exp_name $prefix$dir$dash$model_type \
--dataset_name n3dv2 \
--distortion_loss_w 0.001 \
--opacity_loss_w $loss_w_ \
--entropy_loss_w $loss_w_ \
--num_epochs 60 \
--batch_size 8192 \
--eval_lpips \
--regenerate 0 \
 2>&1 | tee -a $prefix$dir$dash$model_type.log


python train_dynamic.py \
--model_type $model_type \
--root_dir ./data/n3dv/$dir \
--exp_name $prefix$dir$dash$model_type \
--dataset_name n3dv2 \
--distortion_loss_w 0.005 \
--opacity_loss_w $loss_w_ \
--entropy_loss_w $loss_w_ \
--num_epochs 60 \
--batch_size 8192 \
--eval_lpips \
--regenerate 0 \
 2>&1 | tee -a $prefix$dir$dash$model_type.log

python train_dynamic.py \
--model_type $model_type \
--root_dir ./data/n3dv/$dir \
--exp_name $prefix$dir$dash$model_type \
--dataset_name n3dv2 \
--distortion_loss_w 0.01 \
--opacity_loss_w $loss_w_ \
--entropy_loss_w $loss_w_ \
--num_epochs 60 \
--batch_size 8192 \
--eval_lpips \
--regenerate 0 \
 2>&1 | tee -a $prefix$dir$dash$model_type.log

python train_dynamic.py \
--model_type $model_type \
--root_dir ./data/n3dv/$dir \
--exp_name $prefix$dir$dash$model_type \
--dataset_name n3dv2 \
--distortion_loss_w 0.0005 \
--opacity_loss_w $loss_w_ \
--entropy_loss_w $loss_w_ \
--num_epochs 60 \
--batch_size 8192 \
--eval_lpips \
--regenerate 0 \
 2>&1 | tee -a $prefix$dir$dash$model_type.log



done

done