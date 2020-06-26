#!/usr/bin/env bash
export MXNET_ENFORCE_DETERMINISM=1
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export CUDA_VISIBLE_DEVICES=${1}

seed=114
export SEED=${seed}

suffix=
log_interval=10
val_interval=10
validation_start=0

lr_scheduler=cycle
max_lr=3e-3
min_lr=5e-4
total_iter=2508

inc_fraction=.9
cycle_length_decay=.95
cycle_magnitude_decay=.95
lr_factor=.1

optimizer=adam
gid=0,1
num_workers=12
init=xavier

checkpoint_iter=$((total_iter - 9))
growth_rate=4
num_downs=4
loss_type=dice
folder=SecondAttempt

input_size=256
network=drnn
cycle_length=100
growth_rate=4
batch_size=8
base_channel_drnn=8
num_fpg=8
runtime=3e
prefix=${network}
prefix=${prefix}GR${growth_rate}

cmd=`echo train.py -rid ${prefix}_${init}_sd${seed}_CLD${cycle_length_decay}_CMD${cycle_magnitude_decay}_CL${cycle_length}_mxLR${max_lr}_mnLR${min_lr}_bs${batch_size}_fpg${num_fpg}_v${runtime}${suffix} --experiment_name ${folder}/ -gid $gid --batch_size ${batch_size} --l_type ${loss_type} --generator ${network} --num_downs ${num_downs} --num_fpg ${num_fpg} --growth_rate ${growth_rate} --beta1 0.9 --validation_start ${validation_start} --num_workers ${num_workers} --initializer ${init} --min_lr ${min_lr} --max_lr ${max_lr} --cycle_length ${cycle_length} --lr_scheduler ${lr_scheduler} --val_interval ${val_interval} --log_interval ${log_interval} --total_iter ${total_iter} --optimizer ${optimizer} --cycle_length_decay ${cycle_length_decay} --cycle_magnitude_decay ${cycle_magnitude_decay} --inc_fraction ${inc_fraction} --checkpoint_iter ${checkpoint_iter} --lr_factor ${lr_factor} --base_channel_drnn ${base_channel_drnn} --input_size ${input_size}`
echo ${cmd}
python ${cmd}
