export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=0

model_name=CycleNet

root_path_name=./dataset/PEMS/
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS


model_type='linear'
seq_len=96
for pred_len in 12 24 48 96
do
for random_seed in 2025 2026 2027 2028
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
      --enc_in 358 \
      --cycle 2016 \
      --model_type $model_type \
      --train_epochs 20 \
      --patience 3 \
      --gpu $gpu \
      --dropout 0 \
      --use_hour_index 1 \
      --use_day_index 1 \
      --t_dim 16 \
      --s_dim 16 \
      --d_model 128 \
      --d_ff 512 \
      --gpu $gpu \
      --use_revin 0 \
      --itr 1 --batch_size 32 --learning_rate 0.001 --random_seed $random_seed
done
done