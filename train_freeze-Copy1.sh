#!/bin/bash
#./train_freeze.sh /workdir/Desktop/lifelong_learning/lamol_output_freeze/ 

for (( block=6; block<8; block+=2 ))
do
    for (( counter=0; counter<1; counter+=2 ))
    do
        model_dir="${1:0: -1}_block_${block}_seed_$counter${1: -1}"
        python train_freeze_block.py --data_dir /workdir/Desktop/lifelong_learning/lamol_data --model_dir_root ${model_dir} --tasks boolq movie scifact --seq_train_type lll --gen_lm_sample_percentage 0.2 --n_train_epochs 5 --add_task_tokens --seed $counter --layer_to_freeze ${block} --skip_tasks boolq

    done
done


