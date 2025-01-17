# for debugging
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id Exchange_336_192 \
  --model NLinear \
  --des 'Trial' \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 192 \
  --enc_in 8 \
  --itr 1 \
  --train_epochs 10 \
  --batch_size 8  \
  --learning_rate 0.005 \
  --num_workers 0 \
  --individual \
  --log_file logs/trial.log
