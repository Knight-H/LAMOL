#!/bin/bash
#./test_freeze.sh /workdir/Desktop/lifelong_learning/lamol_output_freeze/ 

for (( block=6; block<12; block+=2 ))
do
    for (( counter=0; counter<3; counter++ ))
    do
        model_dir="${1:0: -1}_block_${block}_seed_$counter${1: -1}"
        python test1.py --data_dir /workdir/Desktop/lifelong_learning/lamol_data --model_dir_root $model_dir --tasks boolq movie scifact --seq_train_type lll --gen_lm_sample_percentage 0.2 --n_train_epochs 5 --add_task_tokens 

    done
done


