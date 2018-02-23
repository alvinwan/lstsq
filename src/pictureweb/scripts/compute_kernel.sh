#!/bin/bash

echo "Computing train kernel"

python compute_kernel.py X_train_imagenet_fc7 0.01 --pywren_mode standalone --max_num_jobs 50000

#X_block_scrambled_3_3_pool_12_12_patch_stride_1_1_block_no_normalize_4_29_2017 4.0
#--linear_kernel "XXT(X_block_scrambled_3_3_pool_12_12_patch_stride_1_1_block_no_normalize_04.22.2017)"

echo "Computing test kernel"
python compute_kernel.py X_train_imagenet_fc7 0.01 --test_key X_test_imagenet_fc7 --pywren_mode standalone --max_num_jobs 50000

#X_block_scrambled_3_3_pool_12_12_patch_stride_1_1_block_no_normalize_4_29_2017  4.0 --test_key X_test_scrambled_3_3_pool_12_12_patch_stride_1_1_block_no_normalize_4_29_2017
#--linear_kernel "XYT(X_block_scrambled_3_3_pool_12_12_patch_stride_1_1_block_no_normalize_04.22.2017, X_test_scrambled_3_3_pool_12_12_patch_stride_1_1_block_no_normalize_04.22.2017)"

