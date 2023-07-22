

#python gpu_wait.py




for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_mine9/$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--cuda_ray \
--update_extra_interval 100 \
--num_rays 8192 \
--model_type 1 \
--dt_gamma 0 2>&1 | tee -a trial_mine9$sce.log
done

#for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
#do
#python -u main_dnerf.py data/dnerf/$sce \
# --workspace \
#trial_ash_8192/$sce \
#-O \
#--bound \
#1.0  \
#--scale \
#0.8 \
#--cuda_ray \
#--update_extra_interval 100 \
#--num_rays 8192 \
#--model_type 0 \
#--dt_gamma 0 2>&1 | tee -a trial_ash_8192$sce.log
#done

