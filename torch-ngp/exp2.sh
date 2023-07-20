

#python gpu_wait.py


for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_vannila__6_small/$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--cuda_ray \
--num_rays 2048 \
--model_type 0 \
--iters 60000 \
--dt_gamma 0 2>&1 | tee -a trial_vannila__6_small$sce.log
done

for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_vannila__6_large/$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--cuda_ray \
--num_rays 4096 \
--iters 60000 \
--model_type 0 \
--dt_gamma 0 2>&1 | tee -a trial_vannila__6_large$sce.log
done
