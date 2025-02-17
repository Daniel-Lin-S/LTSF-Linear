if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=336
model_name=DLinear
repeat=3
gpu_id=0
for pred_len in 96 192 336 720; do
  for individual in True False; do
    exp_id="ETTh1_${seq_len}_${pred_len}_ind${individual}"
    log_file="logs/LongForecasting/${model_name}_etth1_${seq_len}_${pred_len}.log"
    command="python -u run_longExp.py \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --data ETTh1 \
    --enc_in 7 \
    --batch_size 32 \
    --learning_rate 0.005 \
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
    --result_file DLinear_result.txt"

    if [ "$individual" = "True" ]; then
      command="$command --individual"
    fi

    eval $COMMAND

    exp_id="ETTh2_${seq_len}_${pred_len}_ind${individual}"
    log_file="logs/LongForecasting/${model_name}_etth2_${seq_len}_${pred_len}.log"
    command="python -u run_longExp.py \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh2.csv \
    --data ETTh2 \
    --enc_in 7 \
    --batch_size 32 \
    --learning_rate 0.005 \
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
    --result_file DLinear_result.txt"

    if [ "$individual" = "True" ]; then
      command="$command --individual"
    fi

    eval $COMMAND

    exp_id="ETTm1_${seq_len}_${pred_len}_ind${individual}"
    log_file="logs/LongForecasting/${model_name}_ettm1_${seq_len}_${pred_len}.log"
    command="python -u run_longExp.py \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm1.csv \
    --data ETTm1 \
    --enc_in 7 \
    --batch_size 8 \
    --learning_rate 0.005 \
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
    --result_file DLinear_result.txt"

    if [ "$individual" = "True" ]; then
      command="$command --individual"
    fi

    eval $COMMAND

    exp_id="ETTm2_${seq_len}_${pred_len}_ind${individual}"
    log_file="logs/LongForecasting/${model_name}_ettm2_${seq_len}_${pred_len}.log"
    command="python -u run_longExp.py \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTm2.csv \
    --data ETTm2 \
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
    --result_file DLinear_result.txt"

    if [ "$individual" = "True" ]; then
      command="$command --individual"
    fi

    eval $COMMAND
  done
done