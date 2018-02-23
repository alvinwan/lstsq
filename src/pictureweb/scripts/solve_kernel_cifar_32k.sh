#!/bin/bash
/home/ubuntu/anaconda3/bin/python solve.py "XXT(coatesng_32768_6_1_15_6_1.0_0(cifar_augmented_train))" \
"cifar/cifar_augmented_train_labels" \
--epochs 6 \
--eval_interval 1 \
--blocks_per_iter 8 \
--test_key  "XYT(coatesng_32768_6_1_15_6_1.0_0(cifar_augmented_train), coatesng_32768_6_1_15_6_1.0_0(cifar_augmented_test))" \
--test_labels_key "cifar/cifar_augmented_test_labels" \
--num_test_blocks 1 \
--lambdav 100 \
--bucket picturewebsolve \
--sheet "augmented_cifar_bcd"
#--start_epoch 2 \
#--start_block 4 \
#--warm_start "/tmp/model.npy" \
#--prev_yhat "/tmp/y_hat.npy"

