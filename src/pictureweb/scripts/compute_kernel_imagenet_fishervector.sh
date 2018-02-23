#!/bin/bash

echo "Computing train kernel"
python compute_kernel.py "X_train_fisher_vector_sharded" --pywren_mode lambda --tasks_per_job 1 --max_num_jobs 50000 --job_max_runtime 600 --in_bucket pictureweb --out_bucket picturewebsolve --type "linear"


echo "Computing test kernel"
python compute_kernel.py "X_train_fisher_vector_sharded" --pywren_mode lambda --tasks_per_job 1 --max_num_jobs 50000 --job_max_runtime 600 --test_key "X_test_fisher_vector_sharded" --in_bucket pictureweb --out_bucket picturewebsolve --type "linear"



