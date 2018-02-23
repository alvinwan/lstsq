#!/bin/bash

echo "Computing train kernel"
python compute_kernel.py "coatesng_1024_6_1_15_6_1.0_0(cifar_train)" --pywren_mode lambda --tasks_per_job 5 --max_num_jobs 50000 --job_max_runtime 9000 --in_bucket pictureweb --out_bucket picturewebsolve --type "linear"


echo "Computing test kernel"
python compute_kernel.py "coatesng_1024_6_1_15_6_1.0_0(cifar_test)" --pywren_mode lambda --tasks_per_job 5 --max_num_jobs 50000 --job_max_runtime 9000 --test_key "coatesng_1024_6_1_15_6_1.0_0(cifar_augmented_test)" --in_bucket pictureweb --out_bucket picturewebsolve --type "linear"



