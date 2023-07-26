

python gpu_wait.py




for seed in 0 1 2 3 42 1337
do
for sce in  "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do


name_='trial_original_ashawkey_setting'

python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
$name_$seed/$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--cuda_ray \
--update_extra_interval 100 \
--num_rays 4096 \
--model_type -1 \
--seed $seed \
--dt_gamma 0 2>&1 | tee -a $name_$seed$sce.log




done
done









