export CUDA_VISIBLE_DEVICES=0,1,2,3
gpu=0
model_name=CycleNet

root_path_name=./dataset/ETT-small/
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2

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
#      --enc_in 7 \
#      --cycle 24 \
#      --model_type $model_type \
#      --train_epochs 30 \
#      --patience 5 \
#      --gpu $gpu \
#      --des 'exp' \
#      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed
#done
#done
#exit
# 下面是加emb加STAR模块的实验
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
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --gpu $gpu \
      --des 'test' \
      --dropout 0 \
      --e_layers 3 \
      --d_model 512 \
      --d_ff 1024 \
      --itr 1 --batch_size 256 --learning_rate 0.001 --random_seed $random_seed
done
done
exit
# 下面是去掉Q拉长L的实验
seq_len=720
for pred_len in 96 #96 192 336 720
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
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 20 \
      --patience 3 \
      --gpu $gpu \
      --des 'NoQ' \
      --dropout 0.2 \
      --d_ff 512 \
      --itr 1 --batch_size 256 --learning_rate 0.0005 --random_seed $random_seed
done
done

for pred_len in 192 #96 192 336 720
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
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 20 \
      --patience 3 \
      --gpu $gpu \
      --des 'NoQ' \
      --dropout 0.2 \
      --d_ff 512 \
      --itr 1 --batch_size 256 --learning_rate 0.0005 --random_seed $random_seed
done
done

for pred_len in 336  #96 192 336 720
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
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 20 \
      --patience 3 \
      --gpu $gpu \
      --des 'NoQ' \
      --dropout 0.2 \
      --d_ff 512 \
      --itr 1 --batch_size 256 --learning_rate 0.0005 --random_seed $random_seed
done
done

for pred_len in 720 #96 192 336 720
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
      --cycle 24 \
      --model_type $model_type \
      --train_epochs 30 \
      --patience 5 \
      --gpu $gpu \
      --des 'NoQ' \
      --dropout 0.0 \
      --itr 1 --batch_size 256 --learning_rate 0.005 --random_seed $random_seed
done
done

