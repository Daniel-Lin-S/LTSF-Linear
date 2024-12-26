# for debugging
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

python -u run_reconExp.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id trial \
  --model VAE \
  --pred_model DLinear \
  --data recon \
  --data_pred custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 8 \
  --des '' \
  --itr 1 \
  --train_epochs 10 \
  --train_pred_epochs 5 \
  --batch_size 8  \
  --learning_rate 0.005 \
  --num_workers 0 \
  --individual >logs/trial_recon.log
