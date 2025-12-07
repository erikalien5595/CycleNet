export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=2

model_name=CycleNet

root_path_name=./dataset/
data_path_name=weather.csv
model_id_name=weather
data_name=custom

model_type='mlp'
seq_len=96
#for pred_len in 96 192 336 720
#do
#for random_seed in 2024 #2025 2026 2027 2028
#do
#    python -u run.py \
#      --is_training 1 \
#      --root_path $root_path_name \
#      --data_path $data_path_name \
#      --model_id $model_id_name'_'$seq_len'_'$pred_len \
#      --model $model_name \
#      --data $data_name \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --enc_in 21 \
#      --cycle 144 \
#      --model_type $model_type \
#      --train_epochs 30 \
#      --patience 5 \
#      --gpu $gpu \
#      --use_hour_index 1 \
#      --use_day_index 1 \
#      --t_dim 16 \
#      --s_dim 16 \
#      --d_model 512 \
#      --d_ff 512 \
#      --des 'test' \
#      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed
#done
#done
# 下面是加embedding且加STAR模块的实验
#for pred_len in 96 192 336 #720
#do
#for random_seed in 2024 #2025 2026 2027 2028
#do
#    python -u run.py \
#      --is_training 1 \
#      --root_path $root_path_name \
#      --data_path $data_path_name \
#      --model_id $model_id_name'_'$seq_len'_'$pred_len \
#      --model $model_name \
#      --data $data_name \
#      --features M \
#      --seq_len $seq_len \
#      --pred_len $pred_len \
#      --enc_in 21 \
#      --cycle 144 \
#      --model_type $model_type \
#      --train_epochs 30 \
#      --patience 5 \
#      --gpu $gpu \
#      --use_hour_index 1 \
#      --use_day_index 1 \
#      --t_dim 16 \
#      --s_dim 16 \
#      --d_model 512 \
#      --d_ff 512 \
#      --e_layers 3 \
#      --des 'test' \
#      --dropout 0.15 \
#      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
#done
#done

for pred_len in 720
do
for random_seed in 2024 #2025 2026 2027 2028
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --cycle 144 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --gpu $gpu \
      --use_hour_index 1 \
      --use_day_index 1 \
      --t_dim 16 \
      --s_dim 16 \
      --d_model 512 \
      --d_ff 512 \
      --e_layers 3 \
      --des 'test' \
      --dropout 0.2 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
done
done



