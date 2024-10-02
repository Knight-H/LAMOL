#!/bin/bash
#./train_freeze.sh  

for (( block=0; block<12; block+=2 ))
do
    for (( counter=0; counter<3; counter+=1 ))
    do
        model_dir="/workdir/Desktop/lifelong_learning/lamol_output_freeze_block_${block}_seed_${counter}/"
        python train_freeze_block.py --data_dir /workdir/Desktop/lifelong_learning/lamol_data --model_dir_root ${model_dir} --tasks boolq movie scifact --seq_train_type lll --gen_lm_sample_percentage 0.2 --n_train_epochs 5 --add_task_tokens --seed $counter --layer_to_freeze ${block} --skip_tasks boolq movie

    done
done


