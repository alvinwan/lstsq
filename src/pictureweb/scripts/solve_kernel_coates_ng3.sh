#!/bin/bash
/home/ubuntu/anaconda3/bin/python solve.py "XXT(coatesng_16384_16_2_49_18_0.916_0(imagenet_train_raw_uint8))" \
"scrambled_train_labels.npy" \
--epochs 10 \
--eval_interval 2 \
--blocks_per_iter 8 \
--test_key  "XYT(coatesng_16384_16_2_49_18_0.916_0(imagenet_train_raw_uint8), coatesng_16384_16_2_49_18_0.916_0(imagenet_test_raw_uint8))" \
--test_labels_key "scrambled_test_labels.npy" \
--lambdav 333 \
--bucket picturewebsolve
