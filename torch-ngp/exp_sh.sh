

python gpu_wait.py


for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_vannila__7_small/$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--cuda_ray \
--model_type 0 \
--dt_gamma 0 2>&1 | tee -a trial_vannila__7_small$sce.log
done



for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_latent_code__7_small/$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--cuda_ray \
--model_type 1 \
--dt_gamma 0 2>&1 | tee -a trial_latent_code__7_small$sce.log
done

