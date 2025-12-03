export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=0

model_name=CycleNet

root_path_name=./dataset/ETT-small/
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1

model_type='mlp'
seq_len=96
for pred_len in 96 192 336 720
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
      --enc_in 7 \
      --cycle 96 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --gpu $gpu \
      --dropout 0 \
      --use_hour_index 1 \
      --use_day_index 1 \
      --t_dim 16 \
      --s_dim 16 \
      --d_model 128 \
      --d_ff 128 \
      --des 'PurePeriodPred' \
      --itr 1 --batch_size 256 --learning_rate 0.0005 --random_seed $random_seed
done
done

