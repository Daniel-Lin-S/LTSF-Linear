# for debugging
if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

for model in NLinear DLinear
do
python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_336_96 \
  --model_recon VAE \
  --model_pred $model \
  --des 'Trial' \
  --data custom \
  --features M \
  --seq_len 336 \
  --pred_len 96 \
  --train_epochs_recon 10 \
  --train_epochs_pred 10 \
  --enc_in 321 \
  --itr 1 \
  --batch_size_recon 8  \
  --batch_size_pred 16  \
  --learning_rate_recon 0.002 \
  --learning_rate_pred 0.005 \
  --individual \
  --num_workers 0 \
  --loss_level latent \
  --config configs/config_vae.yaml \
  --log_file logs/trial_recon.log
done