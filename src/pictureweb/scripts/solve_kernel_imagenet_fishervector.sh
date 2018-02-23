#!/bin/bash
/home/ubuntu/anaconda3/bin/python solve.py "XXT(X_train_fisher_vector_sharded)" \
"fishervector/y_train.npy" \
--epochs 5 \
--eval_interval 1 \
--blocks_per_iter 15 \
--test_key "XYT(X_train_fisher_vector_sharded, X_test_fisher_vector_sharded)" \
--test_labels_key "fishervector/y_test_scrambed.npy" \
--num_test_blocks 1 \
--lambdav 1 \
--bucket picturewebsolve \
--sheet "imagenet_fishervector_bcd"

