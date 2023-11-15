seq_len=24
model_name=PatchTST

root_path_name=./dataset_gluonts/
data_path_name=sol.csv
model_id_name=sol
data_name=custom_gluonts

random_seed=2021
# for pred_len in 96 192 336 720
for pred_len in 24
do
    python -u run_longExp.py \
      --random_seed $random_seed \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name_$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features H \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 137 \
      --e_layers 4 \
      --n_heads 4 \
      --d_model 128 \
      --d_ff 256 \
      --dropout 0.1\
      --fc_dropout 0.2\
      --head_dropout 0\
      --patch_len 4\
      --stride 2\
      --des 'Exp' \
      --train_epochs 100\
      --patience 5\
      --dataset_name $model_id_name \
      --itr 1 --batch_size 32 --learning_rate 0.0001 
done