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
for model_name in FDLinear TFLinear
do 
for pred_len in 96 192 336 720
do
for nfft in 8 16 32
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --exp_id Electricity_$seq_len'_'$pred_len'_nfft'$nfft \
  --model $model_name \
  --nfft $nfft \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 16 \
  --learning_rate 0.005 --individual \
  --log_file logs/STFT/$model_name'_I_'electricity_$seq_len'_'$pred_len.log \
  --result_file result_electricity_stft.txt
done
done
done