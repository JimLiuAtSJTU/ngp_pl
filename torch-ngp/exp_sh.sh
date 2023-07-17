

#for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
#do
#python -u main_dnerf.py data/dnerf/$sce \
# --workspace \
#trial_new_encode/trial_dnerf_hybrid_v3_$sce \
#-O \
#--bound \
#1.0  \
#--scale \
#0.8 \
#--cuda_ray \
#--dt_gamma 0 2>&1 | tee -a vanilla$sce.log
#done


for sce in "bouncingballs" "hellwarrior" "hook" "jumpingjacks" "lego" "mutant" "standup" "trex"
do
python -u main_dnerf.py data/dnerf/$sce \
 --workspace \
trial_new_encode___3/trial_dnerf_hybrid_v3_$sce \
-O \
--bound \
1.0  \
--scale \
0.8 \
--cuda_ray \
--model_type 1 \
--dt_gamma 0 2>&1 | tee -a newmodel$sce.log
done
