#!/bin/bash
#./train_freeze_head.sh /workdir/Desktop/lifelong_learning/lamol_output_freeze_head/ 


for (( counter=2; counter<3; counter+=1 ))
do
    model_dir="${1:0: -1}_seed_$counter${1: -1}"
    python train_freeze_head.py --data_dir /workdir/Desktop/lifelong_learning/lamol_data --model_dir_root ${model_dir} --tasks boolq movie scifact --seq_train_type lll --gen_lm_sample_percentage 0.2 --n_train_epochs 5 --add_task_tokens --seed $counter  --skip_tasks boolq

done


