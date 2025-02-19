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
for nfft in 8 16 32 64; do
    exp_id="Exchange_336_${pred_len}_nfft${nfft}"
    log_file="logs/STFT/${model_name}_I_exchange_336_${pred_len}.log"
    python -u run_longExp.py \
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
    --nfft $nfft \
    --stft_hop_length 2 \
    --log_file $log_file \
    --result_file result_exchange_stft.txt \
    --independent_freqs
done
done
done