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
    exp_id="Exchange_${seq_len}_${pred_len}_ind${individual}"
    log_file="logs/LongForecasting/${model_name}_exchange_${seq_len}_${pred_len}.log"
    command="python -u run_longExp.py \
    --root_path ./dataset/exchange_rate/ \
    --data_path exchange_rate.csv \
    --data custom \
    --enc_in 8 \
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
  done
done