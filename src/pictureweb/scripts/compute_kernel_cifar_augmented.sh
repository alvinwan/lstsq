#!/bin/bash

echo "Computing train kernel"
python compute_kernel.py "coatesng_1024_6_1_15_6_1.0_0(cifar_augmented_train)" --pywren_mode lambda --tasks_per_job 1 --max_num_jobs 50000 --job_max_runtime 600 --in_bucket pictureweb --out_bucket picturewebsolve --type "linear"


echo "Computing test kernel"
python compute_kernel.py "coatesng_1024_6_1_15_6_1.0_0(cifar_augmented_train)" --pywren_mode lambda --tasks_per_job 1 --max_num_jobs 50000 --job_max_runtime 600 --test_key "coatesng_1024_6_1_15_6_1.0_0(cifar_augmented_test)" --in_bucket pictureweb --out_bucket picturewebsolve --type "linear"



