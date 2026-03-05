#!/bin/bash

#SBATCH --partition=gpu_min80gb
#SBATCH --qos=gpu_min80gb

#SBATCH --output=../../_log_slurm/%j.log 

#SBATCH --job-name=loop
nvidia-smi

cd ../../src_clean

target_length_seconds=8
batch_size=64
max_train_step=4000
d_model=32

n_qwen3_features=256

fusion_method=average
drop_prob=0.1
audio_model=qwen2_audio_tower


for seed in 31 32 33 34 35
do
    accuracy_averaging=micro    # Balanced datasets, only micro
    text_model=none             # Only audio datasets, no text model
    for dataset in crema_d ravdess tess
    do

        note="s $seed. $dataset. $accuracy_averaging. A=$audio_model. T=$text_model."
        # note='DEBUG'
        ~/.conda/envs/allgpu/bin/python train.py \
                                                --seed $seed \
                                                --audio_model $audio_model \
                                                --text_model $text_model \
                                                --note "$note" \
                                                --target_length_seconds $target_length_seconds \
                                                --dataset $dataset \
                                                --plot_step_zero \
                                                --accuracy_averaging $accuracy_averaging \
                                                --batch_size $batch_size \
                                                --max_train_step $max_train_step \
                                                --d_model $d_model \
                                                --fusion_method $fusion_method \
                                                --n_qwen3_features $n_qwen3_features \
                                                --test_step 200 \
                                                --drop_prob $drop_prob \
                                                --debug --log_step 4 --test_step 4 --max_test_step 4 --max_train_step 8
    done

    for accuracy_averaging in micro macro
    do  # Unbalanced datasets, use both micro and macro
        for dataset in iemocap meld
        do
            text_model=Qwen3-0.6B  # best model
            note="s $seed. $dataset. $accuracy_averaging. A=$audio_model. T=$text_model."
            # note='DEBUG'
            ~/.conda/envs/allgpu/bin/python train.py \
                                                    --seed $seed \
                                                    --audio_model $audio_model \
                                                    --text_model $text_model \
                                                    --note "$note" \
                                                    --target_length_seconds $target_length_seconds \
                                                    --dataset $dataset \
                                                    --plot_step_zero \
                                                    --accuracy_averaging $accuracy_averaging \
                                                    --batch_size $batch_size \
                                                    --max_train_step $max_train_step \
                                                    --d_model $d_model \
                                                    --fusion_method $fusion_method \
                                                    --n_qwen3_features $n_qwen3_features \
                                                    --drop_prob $drop_prob \
                                                    --test_step 200 \
                                                    --debug --log_step 4 --test_step 4 --max_test_step 4 --max_train_step 8
        done
    done
done