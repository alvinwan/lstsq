#!/bin/bash
/home/ubuntu/anaconda3/bin/python solve.py "rbf(XXT(coatesng_8192_6_1_15_6_1.0_0(cifar_augmented_train)), 0.001)" \
"cifar/cifar_augmented_train_labels" \
--epochs 6 \
--eval_interval 1 \
--blocks_per_iter 8 \
--test_key  "rbf(XYT(coatesng_8192_6_1_15_6_1.0_0(cifar_augmented_train), coatesng_8192_6_1_15_6_1.0_0(cifar_augmented_test)), 0.001)" \
--test_labels_key "cifar/cifar_augmented_test_labels" \
--num_test_blocks 1 \
--lambdav 1e-4 \
--bucket picturewebsolve \
--sheet "augmented_cifar_bcd"

