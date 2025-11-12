export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=1

seq_len=96
model_name=CycleiTransformer
data_name=ETTh1
random_seed=2024
for seq_len in 96 #192 336 512 720
do
for pred_len in 96 #192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id 'ETTh1_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --random_seed $random_seed \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'showcase' \
  --d_model 256 \
  --d_ff 256 \
  --train_epochs 10 \
  --patience 3 \
  --lradj 'type1' \
  --learning_rate 0.0001 \
  --gpu $gpu \
  --itr 1
done
done