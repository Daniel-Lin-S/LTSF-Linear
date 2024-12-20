# for debugging
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/electricity \
  --data_path electricity.csv \
  --model_id Electricity_336_96 \
  --model DLinear \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 321 \
  --des 'Trial' \
  --itr 1 \
  --train_epochs 1 \
  --batch_size 16  \
  --learning_rate 0.005 \
  --num_workers 0 \
  --individual >logs/trial.log
