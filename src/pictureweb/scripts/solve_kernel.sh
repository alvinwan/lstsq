#!/bin/bash
/home/ubuntu/anaconda3/bin/python solve.py "XXT(X_train_3_3_pool_12_12_patch_overlap_stride_1_normalized)" \
"scrambled_train_labels.npy" \
--epochs 3 \
--eval_interval 1 \
--blocks_per_iter 1 \
--test_key "XYT(X_train_3_3_pool_12_12_patch_overlap_stride_1_normalized, X_test_3_3_pool_12_12_patch_overlap_stride_1_normalized)" \
--test_labels_key "scrambled_test_labels.npy" \
--lambdav 1e3
