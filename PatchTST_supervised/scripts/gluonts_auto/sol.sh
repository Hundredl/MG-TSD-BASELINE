seq_len=24
model_name=Autoformer

root_path_name=./dataset_gluonts/
data_path_name=sol.csv
model_id_name=sol
data_name=custom_gluonts

random_seed=2021
input_dim=137
features=H
train_epochs=100

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
    --features $features \
    --seq_len $seq_len \
    --label_len 12 \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in $input_dim \
    --dec_in $input_dim \
    --c_out $input_dim \
    --des 'Exp' \
    --itr 1 \
    --batch_size 32 \
    --learning_rate 0.0001 \
    --dataset_name $model_id_name \
    --train_epochs $train_epochs 
done