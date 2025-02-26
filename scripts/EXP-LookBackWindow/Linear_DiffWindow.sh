model_name=DLinear   # Linear NLinear TFLinear FDLinear

for pred_len in 24 720
do
for seq_len in 48 72 96 120 144 168 192 336 504 672 720
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
    --pred_len $pred_len  \
    --enc_in 321 \
    --des 'Exp' \
    --itr 1 --batch_size 16  --learning_rate 0.001 \
    --log_file logs/LookBackWindow/$model_name'_'electricity_$seq_len'_'$pred_len.log \
    --result_file "${model_name}_result.txt"

  python -u run_longExp.py \
    --is_training 1 \
    --root_path ./dataset/ \
    --data_path ETTh1.csv \
    --exp_id ETTh1_$seq_len'_'$pred_len \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --pred_len $pred_len  \
    --enc_in 7 \
    --des 'Exp' \
    --itr 1 --batch_size 8 \
    --log_file logs/LookBackWindow/$model_name'_'ETTh1_$seq_len'_'$pred_len.log \
    --result_file "${model_name}_result.txt"

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Etth2.csv \
  --exp_id Etth2_$seq_len'_'$pred_len \
  --model $model_name \
  --data Etth2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 \
  --log_file logs/LookBackWindow/$model_name'_'Etth2_$seq_len'_'$pred_len.log \
  --result_file "${model_name}_result.txt"

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path exchange_rate.csv \
  --exp_id Exchange_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 8 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.005 \
  --log_file logs/LookBackWindow/$model_name'_'exchange_rate_$seq_len'_'$pred_len.log \
  --result_file "${model_name}_result.txt"

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path traffic.csv \
  --exp_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 862 \
  --des 'Exp' \
  --itr 1 --batch_size 16 --learning_rate 0.05 \
  --log_file logs/LookBackWindow/$model_name'_'traffic_$seq_len'_'$pred_len.log \
  --result_file "${model_name}_result.txt"

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path weather.csv \
  --exp_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 21 \
  --des 'Exp' \
  --itr 1 --batch_size 16 \
  --log_file logs/LookBackWindow/$model_name'_'weather_$seq_len'_'$pred_len.log \
  --result_file "${model_name}_result.txt"
done
done

for pred_len in 24 720
do
for seq_len in 36 48 60 72 144 288
do
  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Ettm1.csv \
  --exp_id Ettm1_$seq_len'_'$pred_len \
  --model $model_name \
  --data Ettm1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 8 --learning_rate 0.0001 \
  --log_file logs/LookBackWindow/$model_name'_'Ettm1_$seq_len'_'$pred_len.log \
  --result_file "${model_name}_result.txt"

  python -u run_longExp.py \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path Ettm2.csv \
  --exp_id Ettm2_$seq_len'_'$pred_len \
  --model $model_name \
  --data Ettm2 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len  \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 \
  --log_file logs/LookBackWindow/$model_name'_'Ettm2_$seq_len'_'$pred_len.log \
  --result_file "${model_name}_result.txt"
done
done

for pred_len in 24 60
do
for seq_len in 26 52 78 104 130 156 208
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
  --pred_len $pred_len  \
  --enc_in 7 \
  --des 'Exp' \
  --itr 1 --batch_size 32 --learning_rate 0.05 \
  --log_file logs/LookBackWindow/$model_name'_'ili_$seq_len'_'$pred_len.log \
  --result_file "${model_name}_result.txt"
done
done