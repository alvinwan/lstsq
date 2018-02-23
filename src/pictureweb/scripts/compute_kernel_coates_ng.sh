#!/bin/bash

echo "Computing train kernel"
python compute_kernel.py "coatesng_16384_16_2_49_18_0.916_0(imagenet_train_raw_uint8)" --pywren_mode standalone --tasks_per_job 5 --max_num_jobs 50000 --job_max_runtime 9000 --in_bucket picturewebhyperband --out_bucket picturewebsolve --type "linear" --local



echo "Computing test kernel"
python compute_kernel.py "coatesng_16384_16_2_49_18_0.916_0(imagenet_train_raw_uint8)" --pywren_mode standalone --tasks_per_job 5 --max_num_jobs 50000 --job_max_runtime 9000 --test_key "coatesng_16384_16_2_49_18_0.916_0(imagenet_test_raw_uint8)" --in_bucket picturewebhyperband --out_bucket picturewebsolve --type "linear"



