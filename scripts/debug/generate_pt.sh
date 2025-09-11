#!/bin/bash

source /ssd1/muntasir/Desktop/AutoTimes/.autotimes/bin/activate

python preprocess.py \
    --gpu 0 \
    --llm_ckp_dir /ssd1/muntasir/Desktop/AutoTimes/llama-7b \
    --dataset permno10000