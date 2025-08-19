#!/bin/bash

python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id PANELCOV_672_96 \
  --model AutoTimes_Llama \
  --data panel_cov \
  --root_path ./dataset/panel \
  --data_path panel_cleaned.csv \
  --seq_len 36 --label_len 32 --token_len 4 \
  --test_seq_len 36 --test_label_len 32 --test_pred_len 4 \
  --batch_size 2 --learning_rate 1e-5 --train_epochs 3 \
  --panel_id_col PERMNO --panel_time_col DATE --panel_y_col actual \
  --panel_cov_cols "mean,std,capital_ratio,equity_invcap,debt_invcap,totdebt_invcap,at_turn,pay_turn,rect_turn,sale_equity,sale_invcap,invt_act,rect_act,ocf_lct,cash_debt,cash_lt,cfm,short_debt,profit_lct,curr_debt,debt_ebitda,dltt_be,lt_debt,lt_ppent,cash_ratio,curr_ratio,quick_ratio,accrual,rd_sale,adv_sale,staff_sale,GProf,aftret_eq,aftret_equity,aftret_invcapx,gpm,npm,opmad,opmbd,pretret_earnat,pretret_noa,ptpm,roa,roce,roe,de_ratio,debt_assets,debt_at,debt_capital,bm,CAPEI,evm,pe_exi,pe_inc,pe_op_basic,ps,ptb" \
  --target_var_idx -1 \
  --llm_ckp_dir /ssd1/muntasir/Desktop/AutoTimes/llama-7b \
  --gpu 1 \