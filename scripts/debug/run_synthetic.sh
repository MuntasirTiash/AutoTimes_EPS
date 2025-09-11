#!/bin/bash

python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id permno10000 \
  --model AutoTimes_Llama \
  --data custom \
  --root_path ./dataset/custom/ \
  --data_path permno10000.csv \
  --test_data_path permno10000.csv \
  --checkpoints ./checkpoints/ \
  --seq_len 36 \
  --label_len 32 \
  --token_len 4 \
  --test_seq_len 36 \
  --test_label_len 32 \
  --test_pred_len 4 \
  --drop_last \
  --drop_short \
  --train_epochs 20 \
  --batch_size 4 \
  --learning_rate 0.0001 \
  --des test_run \
  --llm_ckp_dir /ssd1/muntasir/Desktop/AutoTimes/llama-7b \
  --gpu 0 \

