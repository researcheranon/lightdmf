#!/bin/bash

#SBATCH --partition=gpu_min80gb
#SBATCH --qos=gpu_min80gb

#SBATCH --output=../../_log_slurm/%j.log 

#SBATCH --job-name=case
nvidia-smi

cd ../../src
datasets_to_merge="crema_d iemocap meld ravdess tess goemotions"

target_length_seconds=8
batch_size=64
max_train_step=8000
d_model=32

n_qwen3_features=256

fusion_method=attention
drop_prob=0.1

dataset=merged
accuracy_averaging=micro

for seed in 31 32 33 34 35
do

    audio_model=qwen2_audio_tower  # best audio model
    for text_model in  Qwen3-0.6B llama minilm
    do  # all text models
        note="seed $seed. case mapping. A=$audio_model. T=$text_model."

        # echo "Processing: $datasets_to_merge"
        ~/.conda/envs/allgpu/bin/python train.py \
            --seed $seed \
            --audio_model $audio_model \
            --text_model $text_model \
            --target_length_seconds $target_length_seconds \
            --dataset $dataset \
            --datasets_to_merge $datasets_to_merge \
            --plot_step_zero \
            --accuracy_averaging $accuracy_averaging \
            --batch_size $batch_size \
            --max_train_step $max_train_step \
            --d_model $d_model \
            --fusion_method $fusion_method \
            --exhaustive_test \
            --multi_classifier \
            --n_qwen3_features $n_qwen3_features \
            --drop_prob $drop_prob \
            --test_step 200 \
            --case_mapping \
            --skip_test \
            --note "$note" \
            --use_preextracted_case_features \
            --debug --log_step 4 --test_step 4 --max_test_step 4 --max_train_step 8        
    done

    text_model=Qwen3-0.6B  # best text model
    for audio_model in wav2vec2_xls distil_whisper
    do  # all audio models
        note="seed $seed. case mapping. A=$audio_model. T=$text_model."

        # echo "Processing: $datasets_to_merge"
        ~/.conda/envs/allgpu/bin/python train.py \
            --seed $seed \
            --audio_model $audio_model \
            --text_model $text_model \
            --target_length_seconds $target_length_seconds \
            --dataset $dataset \
            --datasets_to_merge $datasets_to_merge \
            --plot_step_zero \
            --accuracy_averaging $accuracy_averaging \
            --batch_size $batch_size \
            --max_train_step $max_train_step \
            --d_model $d_model \
            --fusion_method $fusion_method \
            --exhaustive_test \
            --multi_classifier \
            --n_qwen3_features $n_qwen3_features \
            --drop_prob $drop_prob \
            --test_step 200 \
            --case_mapping \
            --skip_test \
            --note "$note" \
            --use_preextracted_case_features \
            --debug --log_step 4 --test_step 4 --max_test_step 4 --max_train_step 8        
    done

    # lightest models: distil_whisper and minilm
    audio_model=distil_whisper  # lightest audio model
    text_model=minilm  # lightest text model
    note="seed $seed. case mapping. A=$audio_model. T=$text_model."

    # echo "Processing: $datasets_to_merge"
    ~/.conda/envs/allgpu/bin/python train.py \
        --seed $seed \
        --audio_model $audio_model \
        --text_model $text_model \
        --target_length_seconds $target_length_seconds \
        --dataset $dataset \
        --datasets_to_merge $datasets_to_merge \
        --plot_step_zero \
        --accuracy_averaging $accuracy_averaging \
        --batch_size $batch_size \
        --max_train_step $max_train_step \
        --d_model $d_model \
        --fusion_method $fusion_method \
        --exhaustive_test \
        --multi_classifier \
        --n_qwen3_features $n_qwen3_features \
        --drop_prob $drop_prob \
        --test_step 200 \
        --case_mapping \
        --skip_test \
        --note "$note" \
        --use_preextracted_case_features \
        --debug --log_step 4 --test_step 4 --max_test_step 4 --max_train_step 8        

    
    # ablation: average fusion instead of attention
    # lightest models: distil_whisper and minilm
    audio_model=distil_whisper  # lightest audio model
    text_model=minilm  # lightest text model
    note="seed $seed. case mapping. average fusion. A=$audio_model. T=$text_model."

    ~/.conda/envs/allgpu/bin/python train.py \
        --seed $seed \
        --audio_model $audio_model \
        --text_model $text_model \
        --target_length_seconds $target_length_seconds \
        --dataset $dataset \
        --datasets_to_merge $datasets_to_merge \
        --plot_step_zero \
        --accuracy_averaging $accuracy_averaging \
        --batch_size $batch_size \
        --max_train_step $max_train_step \
        --d_model $d_model \
        --fusion_method average \
        --exhaustive_test \
        --multi_classifier \
        --n_qwen3_features $n_qwen3_features \
        --drop_prob $drop_prob \
        --test_step 200 \
        --case_mapping \
        --skip_test \
        --note "$note" \
        --use_preextracted_case_features \

done
