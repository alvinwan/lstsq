#!/bin/bash
/home/ubuntu/anaconda3/bin/python solve.py "XXT(X_train_imagenet_fc7)" \
"y_train_fc7.npy" \
--epochs 3 \
--eval_interval 1 \
--blocks_per_iter 1 \
--test_key "XYT(X_train_imagenet_fc7, X_test_imagenet_fc7)" \
--test_labels_key "y_val_fc7.npy" \
--lambdav 1e3
