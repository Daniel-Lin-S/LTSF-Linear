#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/STFT" ]; then
    mkdir ./logs/STFT
fi


repeat=3
gpu_id=0
for pred_len in 96 192 336 720; do
for individual in False True; do
for independent_freqs in False True; do
    exp_id="Exchange_336_${pred_len}_ind${individual}_freqind${independent_freqs}"
    log_file="logs/STFT/${model_name}_I_exchange_336_${pred_len}.log"
    COMMAND="python -u run_longExp.py \
        --is_training 1 \
        --root_path ./dataset/exchange_rate \
        --data_path exchange_rate.csv \
        --exp_id $exp_id \
        --model TFLinear \
        --gpu $gpu_id \
        --des 'Exp' \
        --data custom \
        --features M \
        --seq_len 336 \
        --pred_len $pred_len \
        --enc_in 8 \
        --itr $repeat \
        --train_epochs 10 \
        --batch_size 8 \
        --learning_rate 0.005 \
        --nfft 4 \
        --stft_hop_length 2 \
        --log_file $log_file \
        --result_file result_exchange_stft.txt"

    if [ "$individual" = "True" ]; then
        COMMAND="$COMMAND --individual"
    fi

    if [ "$independent_freqs" = "True" ]; then
        COMMAND="$COMMAND --independent_freqs"
    fi

    # Run the command
    eval $COMMAND
done
done
done