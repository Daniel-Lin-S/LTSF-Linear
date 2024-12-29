# for debugging
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

python -u run_reconExp.py \
  --is_training 1 \
  --root_path ./dataset/exchange_rate \
  --data_path exchange_rate.csv \
  --model_id trial \
  --model_recon VAE \
  --model_pred DLinear \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --enc_in 8 \
  --loss_level origin \
  --itr 1 \
  --train_epochs_recon 20 \
  --train_epochs_pred 5 \
  --patience_recon 5 \
  --patience_pred 3 \
  --batch_size_recon 8  \
  --batch_size_pred 16  \
  --learning_rate_recon 0.001 \
  --learning_rate_pred 0.005 \
  --num_workers 0 \
  --individual >logs/trial_recon_0beta_pred.log
