#!/bin/bash
#./train_freeze.sh /workdir/Desktop/lifelong_learning/lamol_output/ 

for (( counter=0; counter<3; counter++ ))
do
    model_dir="${1:0: -1}_baseline_seed_$counter${1: -1}"
    python train-start-at-task2.py --data_dir /workdir/Desktop/lifelong_learning/lamol_data --model_dir_root $model_dir --tasks boolq movie scifact --seq_train_type lll --gen_lm_sample_percentage 0.2 --n_train_epochs 5 --add_task_tokens --seed $counter --skip_tasks boolq

done



