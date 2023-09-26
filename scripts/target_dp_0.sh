#!/bin/bash

# Loading the required module
#source /etc/profile
#module load anaconda/2021a

export PYTHONNOUSERSITE=True    # prevent using packages from base
#source activate th102_cu113_tgconda

python main_mp.py \
    --output-dir 'models/mp/dp_equiformer/se_l2/target@0/' \
    --model-name 'dot_product_attention_transformer_l2' \
    --input-irreps '100x0e' \
    --target 0 \
    --data-path 'datasets/mp' \
    --feature-type 'crystalnet' \
    --batch-size 256 \
    --radius 5.0 \
    --num-basis 128 \
    --drop-path 0.0 \
    --weight-decay 5e-3 \
    --lr 5e-4 \
    --min-lr 1e-6 \
    --no-model-ema \
    --no-amp
