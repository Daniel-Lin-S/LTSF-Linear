python -u run_reconExp.py \
  --is_training 1 \
  --root_path ./dataset/weather \
  --data_path weather.csv \
  --model_id trial \
  --model VAE \
  --pred_model DLinear \
  --data recon \
  --data_pred custom \
  --features M \
  --seq_len 336 \
  --pred_len 336 \
  --enc_in 21 \
  --des '' \
  --itr 1 \
  --train_epochs 10 \
  --train_pred_epochs 1 \
  --batch_size 16  \
  --learning_rate 0.005 \
  --num_workers 0 \
  --individual >logs/trial_recon_weather.log