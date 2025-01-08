# for debugging
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

python -u run_reconExp.py \
  --is_training 3 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_336_96 \
  --model_recon VAE \
  --model_pred DLinear \
  --des 'Trial' \
  --data ETTh1 \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 7 \
  --itr 1 \
  --batch_size_recon 16  \
  --batch_size_pred 32  \
  --learning_rate_recon 0.001 \
  --learning_rate_pred 0.005 \
  --individual \
  --num_workers 0 \
  --log_file logs/trial_recon_0beta_recon.log
