#!/bin/bash
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=104
model_name=DLinear
repeat=3
gpu_id=0
for pred_len in 24 36 48 60; do
  for individual in True False; do
    exp_id="Illness_${seq_len}_${pred_len}_ind${individual}"
    log_file="logs/LongForecasting/${model_name}_illness_${seq_len}_${pred_len}.log"
    COMMAND="python -u run_longExp.py \
    --root_path ./dataset/illness/ \
    --data_path national_illness.csv \
    --data custom \
    --enc_in 7 \
    --batch_size 32 \
    --learning_rate 0.01 \
    --exp_id $exp_id \
    --model $model_name \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len \
    --is_training 1 \
    --des 'Exp' \
    --gpu $gpu_id \
    --itr $repeat \
    --log_file $log_file \
    --result_file DLinear_result.csv"

    if [ "$individual" = "True" ]; then
      COMMAND="$COMMAND --individual"
    fi

    eval $COMMAND
  done
done