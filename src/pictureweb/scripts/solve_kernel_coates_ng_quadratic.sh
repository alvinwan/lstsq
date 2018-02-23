#!/bin/bash
/home/ubuntu/anaconda3/bin/python solve.py "ColumnSharded(quadratic(XXT(coatesng_16384_12_1_109_68_0.6079_0(imagenet_train_raw_uint8)))_retry, 1.5e-07), 16)" \
"scrambled_train_labels.npy" \
--epochs 10 \
--eval_interval 1 \
--blocks_per_iter 3750 \
--test_key  "quadratic(XYT(coatesng_16384_12_1_109_68_0.6079_0(imagenet_train_raw_uint8), coatesng_16384_12_1_109_68_0.6079_0(imagenet_test_raw_uint8))_retry, 1.5e-07)" \
--test_labels_key "scrambled_test_labels.npy" \
--lambdav 1e-6 \
--bucket picturewebsolve
#--start_block 17 \
#--warm_start "/tmp/models/ColumnSharded(quadratic(XXT(coatesng_16384_12_1_109_68_0.6079_0(imagenet_train_raw_uint8)))_retry, 1.5e-07), 16)/lambdav_1e-06_trt1_0.5102332482806691_trt5_0.5998687134464126_tt1_0.222412109375_tt5_0.34130859375_epoch_0_block_7.model.npy" \
#--prev_yhat "/tmp/models/ColumnSharded(quadratic(XXT(coatesng_16384_12_1_109_68_0.6079_0(imagenet_train_raw_uint8)))_retry, 1.5e-07), 16)/lambdav_1e-06_trt1_0.5102332482806691_trt5_0.5998687134464126_tt1_0.222412109375_tt5_0.34130859375_epoch_0_block_7.y_train_hat.npy"
