#!/bin/bash

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/STFT" ]; then
    mkdir ./logs/STFT
fi

seq_len=336
repeat=3
gpu_id=0
stft_hop_lengths=(4 8 16)
nffts=(8 16 32)
for model_name in FDLinear STFTLinear
do 
for pred_len in 96 192 336 720
do
for i in "${!nffts[@]}"
do
nfft="${nffts[$i]}"
stft_hop_length="${stft_hop_lengths[$i]}"
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --stft_hop_length $stft_hop_length \
  --nfft $nfft \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 8 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 8 --learning_rate 0.005 --individual \
  --log_file logs/STFT/$model_name'_I_'exchange_$seq_len'_'$pred_len.log 
done
done
done