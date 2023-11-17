seq_len=24
model_name=TimeDiff

root_path_name=./dataset_gluonts/
data_path_name=sol.csv
model_id_name=sol
data_name=custom_gluonts

random_seed=802
input_dim=137
features=H
train_epochs=40
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
    --batch_size 256\
    --ddpm_layers_inp 5\
    --ddpm_layers_I 5\
    --ddpm_layers_II 5\
    --cond_ddpm_num_layers 5 \
    --interval 1000 \
    --ddpm_dim_diff_steps 100 \
    --learning_rate 0.001 
done