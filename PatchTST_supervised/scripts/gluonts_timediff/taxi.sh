seq_len=24
model_name=TimeDiff

root_path_name=./dataset_gluonts/
data_path_name=taxi.csv
model_id_name=taxi
data_name=custom_gluonts

random_seed=1086
input_dim=1214
features=H
train_epochs=100
pretrain_epochs=20

for pred_len in 24
do
  python -u run_longExp.py \
    --random_seed $random_seed \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}_${seq_len}_$pred_len \
    --model $model_name \
    --data $data_name \
    --dataset_name $model_id_name \
    --features $features \
    --seq_len $seq_len \
    --label_len 12 \
    --pred_len $pred_len \
    --train_epochs $train_epochs \
    --pretrain_epochs $pretrain_epochs \
    --batch_size 64
done