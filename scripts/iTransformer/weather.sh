export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=0

model_name=CycleiTransformer

seq_len=96
data_name=custom
random_seed=2023
for seq_len in 96 #192 336 512 720
do
for pred_len in 96 192 336 720
do
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id 'Weather_'$seq_len'_'$pred_len \
  --model $model_name \
  --data $data_name \
  --random_seed $random_seed \
  --features M \
  --revin 1 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model 512 \
  --d_ff 512 \
  --dropout 0.1\
  --batch_size 32 \
  --gpu $gpu \
  --train_epochs 30 \
  --patience 5 \
  --lradj 'type1'\
  --learning_rate 0.0001 \
  --itr 1 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done
done





