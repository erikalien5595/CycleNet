export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=0

model_name=CycleiTransformer
root_path_name=./dataset/Solar/
data_path_name=solar_AL.txt
model_id_name=Solar
data_name=Solar
enc_in=137
seq_len=96
#lradj="TST"
for random_seed in 2024 #2025 2023
do
for pred_len in 96 192 336 720
do
python -u run.py \
  --is_training 1 \
  --random_seed $random_seed \
  --model $model_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name'_'$seq_len'_'$pred_len \
  --data $data_name \
  --features M \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in $enc_in \
  --e_layers 2 \
  --d_model 512 \
  --d_ff 512 \
  --batch_size 32 \
  --learning_rate 0.0005 \
  --lradj 'type1' \
  --dropout 0.1 \
  --train_epochs 30 \
  --patience 3 \
  --des 'test' \
  --gpu 0 \
  --itr 1
done
done
exit
