#!/bin/bash
/home/ubuntu/anaconda3/bin/python solve.py "XXT(coatesng_16384_10_1_100_49_0.6311_0(imagenet_train_raw_uint8))" \
"scrambled_train_labels.npy" \
--epochs 3 \
--eval_interval 1 \
--blocks_per_iter 1 \
--test_key  "XYT(coatesng_16384_10_1_100_49_0.6311_0(imagenet_train_raw_uint8), coatesng_16384_10_1_100_49_0.6311_0(imagenet_test_raw_uint8))" \
--test_labels_key "scrambled_test_labels.npy" \
--lambdav 83.0 \
--bucket picturewebsolve \
--start_block 7

