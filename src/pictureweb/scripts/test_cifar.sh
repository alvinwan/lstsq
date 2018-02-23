#!/bin/bash
/home/ubuntu/anaconda3/bin/python test.py \
/tmp/model.npy \
"rbf(XYT(coatesng_8192_6_1_15_6_1.0_0(cifar_augmented_train), coatesng_8192_6_1_15_6_1.0_0(cifar_augmented_test)), 0.001)" \
"cifar/cifar_test_labels" \
--augmented_test_idxs "cifar/cifar_test_augment_idxs" \
--bucket picturewebsolve \
--augmented

