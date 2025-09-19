# run.py
import argparse
import os
import torch
import numpy as np
import random
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

def fix_seeds(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser(description='AutoTimes + LLaMA with text fusion')

    # ---------------- core data/model args ----------------
    parser.add_argument('--model', type=str, default='AutoTimes_Llama')
    parser.add_argument('--data', type=str, default='panel_cov')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)

    parser.add_argument('--seq_len', type=int, default=36)
    parser.add_argument('--label_len', type=int, default=32)
    parser.add_argument('--token_len', type=int, default=4)

    parser.add_argument('--test_seq_len', type=int, default=36)
    parser.add_argument('--test_label_len', type=int, default=32)
    parser.add_argument('--test_pred_len', type=int, default=4)

    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--train_epochs', type=int, default=10)
    parser.add_argument('--tmax', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--mlp_hidden_layers', type=int, default=2)
    parser.add_argument('--mlp_activation', type=str, default='gelu')

    parser.add_argument('--use_amp', action='store_true', default=False)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='cuda:0')

    parser.add_argument('--cosine', action='store_true', default=False)
    parser.add_argument('--checkpoints', type=str, default='./checkpoints')
    parser.add_argument('--visualize', action='store_true', default=False)
    parser.add_argument('--patience', type=int, default=5, help='epochs with no improvement before early stop')

    # ---------------- panel_cov specific ----------------
    parser.add_argument('--panel_id_col', type=str, default='PERMNO')
    parser.add_argument('--panel_time_col', type=str, default='DATE')
    parser.add_argument('--panel_y_col', type=str, default='actual')
    parser.add_argument('--panel_cov_cols', type=str, default='')  # comma-separated; empty -> infer
    parser.add_argument('--drop_short', action='store_true', default=False)
    parser.add_argument('--seasonal_patterns', type=str, default=None)
    
    # ---------------- LLaMA backbone ----------------
    parser.add_argument('--llama_model_name', type=str, default='/ssd1/muntasir/Desktop/AutoTimes/llama-7b')
    parser.add_argument('--llama_dtype', type=str, default='float32')  # float32|float16|bfloat16
    parser.add_argument('--freeze_llama', action='store_true', default=True)
    parser.add_argument('--llama_grad_ckpt', action='store_true', default=False)
    parser.add_argument('--hidden_dim_of_gpt2', type=int, default=4096)  # keep equal to LLaMA-7B

    # ---------------- Text fusion ----------------
    parser.add_argument('--use_text', action='store_true', default=False)
    parser.add_argument('--text_mode', type=str, default='emb', choices=['emb', 'ids'])
    parser.add_argument('--text_dim', type=int, default=4096)            # for 'emb' mode
    parser.add_argument('--text_index_csv', type=str, default=None)      # PERMNO,FILING_DATE,EMB_PATH
    parser.add_argument('--text_ids_index_csv', type=str, default=None)  # PERMNO,FILING_DATE,NPZ_PATH

    # ---------------- test/infer ----------------
    parser.add_argument('--test_dir', type=str, default=None)
    parser.add_argument('--test_file_name', type=str, default=None)

    args = parser.parse_args()
    fix_seeds(2024)

    # ------------- derive token_num for marks -------------
    args.token_num = args.seq_len // max(1, args.token_len)

    # ------------- pack panel_cov options for data_provider -------------
    # data_provider reads args.* directly; ensure they exist
    args.panel_text_index_csv = args.text_index_csv if args.text_mode == 'emb' else args.text_ids_index_csv
    args.panel_text_mode = args.text_mode
    args.panel_text_dim = args.text_dim

    # ------------- run experiment -------------
    setting = f"long_term_forecast_PANELCOV_{args.seq_len}_{args.token_len}_{args.model}"
    exp = Exp_Long_Term_Forecast(args)
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

if __name__ == '__main__':
    main()