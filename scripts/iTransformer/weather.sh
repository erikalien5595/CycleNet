export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=0

if [ ! -d "./logs" ]; then
    mkdir ./logs
fi

if [ ! -d "./logs/LongForecasting" ]; then
    mkdir ./logs/LongForecasting
fi
seq_len=96
model_name=iTransformer
data_name=custom
random_seed=2023
for pred_len in 96 192 336 720
do
    python -u run.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path ./dataset/weather/ \
      --data_path weather.csv \
      --model_id Weather'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --revin 0 \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 21 \
      --e_layers 3 \
      --d_model 512 \
      --d_ff 512 \
      --dropout 0.1\
      --des 'corr_constraint' \
      --train_epochs 10 \
      --patience 3\
      --lradj 'type1'\
      --gpu ${gpu} \
      --itr 1 --batch_size 32 --learning_rate 0.0001 #>logs/LongForecasting/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log
done