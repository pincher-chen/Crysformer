#!/bin/bash

# Loading the required module
#source ~/script/cnmodule_2.sh

#module load  anaconda3/5.3.1
#module load  CUDA/11.1.0
#module load gcc/9.2.0

#conda activate equiformer2
#module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
#source activate th102_cu113_tgconda

CUDA_VISIBLE_DEVICES=0 python -u main_mp.py \
    --output-dir 'models/mp/target_0/' \
    --model-name 'graph_attention_transformer_nonlinear_l2_e3' \
    --input-irreps '100x0e' \
    --target 0 \
    --data-path 'datasets/mp_py_cut12' \
    --feature-type 'crystalnet' \
    --batch-size 128 \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --weight-decay 1e-3 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp  \
    --loss l1

