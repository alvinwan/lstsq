#!/bin/bash
/data/vaishaal/anaconda3/bin/python solve.py "ColumnSharded(XXT(mnist_train_1), 16)" \
"y_train_mnist.npy" \
--epochs 3 \
--eval_interval 1 \
--blocks_per_iter 256 \
--test_key "XYT(mnist_train_64, mnist_test_64)" \
--test_labels_key "y_test_mnist.npy" \
--lambdav 1e-8 \
--start_block 14
