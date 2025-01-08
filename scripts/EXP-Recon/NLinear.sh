if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/TwoStageForecasting" ]; then
    mkdir ./logs/TwoStageForecasting
fi
seq_len=336
model_recon=VAE
model_pred=NLinear
repeat=3
for pred_len in 96 192 336 729
do
python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id Electricity_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 8 \
  --batch_size_pred 16 \
  --learning_rate_recon 0.001 \
  --learning_rate 0.005 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'electricity_$seq_len'_'$pred_len.log 

python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 8 \
  --batch_size_pred 16 \
  --learning_rate_recon 0.001 \
  --learning_rate 0.005 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'traffic_$seq_len'_'$pred_len.log

python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 8 \
  --batch_size_pred 16 \
  --learning_rate_recon 0.001 \
  --learning_rate 0.005 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'weather_$seq_len'_'$pred_len.log

python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/exchange_rate/ \
  --data_path exchange_rate.csv \
  --model_id Exchange_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 8 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 8 \
  --batch_size_pred 8 \
  --learning_rate_recon 0.001 \
  --learning_rate 0.005 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'exchange_$seq_len'_'$pred_len.log 

python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 16 \
  --batch_size_pred 32 \
  --learning_rate_recon 0.001 \
  --learning_rate 0.005 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'ETTh1_$seq_len'_'$pred_len.log

python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 16 \
  --batch_size_pred 32 \
  --learning_rate_recon 0.001 \
  --learning_rate 0.005 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'ETTh2_$seq_len'_'$pred_len.log 

python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 16 \
  --batch_size_pred 32 \
  --learning_rate_recon 0.001 \
  --learning_rate 0.005 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'ETTm1_$seq_len'_'$pred_len.log 

python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id ETTm2_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 16 \
  --batch_size_pred 32 \
  --learning_rate_recon 0.002 \
  --learning_rate 0.01 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'ETTm2_$seq_len'_'$pred_len.log 
done

seq_len=104
for pred_len in 24 36 48 60
do
python -u run_reconExp.py \
  --is_training 2 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id national_illness_$seq_len'_'$pred_len \
  --model_recon $model_recon \
  --model_pred $model_pred \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 18 \
  --enc_in 7 \
  --des 'Exp' \
  --itr $repeat \
  --batch_size_recon 16 \
  --batch_size_pred 32 \
  --learning_rate_recon 0.002 \
  --learning_rate 0.01 --individual \
  --log_file logs/TwoStageForecasting/$model_name'_'ILI_$seq_len'_'$pred_len.log
done