#!/bin/bash
#./test_freeze.sh


for (( counter=0; counter<3; counter++ ))
do
    model_dir="/root/LAMOL/github/LAMOL/lamol_output_freeze_head_seed_$counter${1: -1}/"
    python test.py --data_dir /root/LAMOL/lamol_data --model_dir_root $model_dir --tasks boolq movie scifact --seq_train_type lll --gen_lm_sample_percentage 0.2 --n_train_epochs 5 --add_task_tokens 

done



