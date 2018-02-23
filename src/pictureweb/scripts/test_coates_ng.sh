#!/bin/bash
/home/ubuntu/anaconda3/bin/python test.py \
"/tmp/quadratic(XXT(coatesng_16384_12_1_109_68_0.6079_0(imagenet_train_raw_uint8)))_retry, 1.5e-07)_epochs_5_train_acc_0.9903049329244353_norm_722255111.6259025.model.npy" \
"quadratic(XYT(coatesng_16384_12_1_109_68_0.6079_0(imagenet_train_raw_uint8), coatesng_16384_12_1_109_68_0.6079_0(imagenet_test_raw_uint8))_retry, 1.5e-07)" \
"scrambled_test_labels.npy" \
--bucket picturewebsolve \
