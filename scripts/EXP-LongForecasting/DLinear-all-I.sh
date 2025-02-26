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
for pred_len in 96 192 336 720
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path electricity.csv \
  --exp_id Electricity_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 321 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 16  --learning_rate 0.005 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'electricity_$seq_len'_'$pred_len.log \
  --result_file DLinear_result.csv

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --exp_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 862 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 16 --learning_rate 0.005 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'traffic_$seq_len'_'$pred_len.log \
  --result_file DLinear_result.csv

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --exp_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 21 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 16 --learning_rate 0.005 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'weather_$seq_len'_'$pred_len.log \
  --result_file DLinear_result.csv

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --exp_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 8 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 8 --learning_rate 0.005 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'exchange_$seq_len'_'$pred_len.log \
  --result_file DLinear_result.csv

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --exp_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 32 --learning_rate 0.005 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'ETTh1_$seq_len'_'$pred_len.log \
  --result_file DLinear_result.csv

# if pred_len=336, lr=0.001; if pred_len=720, lr=0.0001
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --exp_id ETTh2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 32 --learning_rate 0.005 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'ETTh2_$seq_len'_'$pred_len.log \
  --result_file DLinear_result.csv

# if pred_len=336, lr=0.005; if pred_len=720, lr=0.0005
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --exp_id ETTm1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 8 --learning_rate 0.005 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'ETTm1_$seq_len'_'$pred_len.log \
  --result_file DLinear_result.csv

python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --exp_id ETTm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 32 --learning_rate 0.01 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'ETTm2_$seq_len'_'$pred_len.log  \
  --result_file DLinear_result.csv
done

seq_len=104
for pred_len in 24 36 48 60
do
python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path national_illness.csv \
  --exp_id national_illness_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 18 \
  --pred_len $pred_len \
  --enc_in 7 \
  --des 'Exp' \
  --gpu $gpu_id \
  --itr $repeat --batch_size 32 --learning_rate 0.01 --individual \
  --log_file logs/LongForecasting/$model_name'_I_'ILI_$seq_len'_'$pred_len.log \
  --result_file DLinear_result.csv
done

