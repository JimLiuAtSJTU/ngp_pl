

#python gpu_wait.py


for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_mine2/$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--cuda_ray \
--update_extra_interval 100 \
--num_rays 4096 \
--model_type 1 \
--dt_gamma 0 2>&1 | tee -a trial_mine2$sce.log
done

