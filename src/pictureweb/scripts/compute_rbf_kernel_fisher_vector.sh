#!/bin/bash

echo "Computing train kernel"
python compute_kernel.py "X_train_fisher_vector_sharded" --pywren_mode lambda --tasks_per_job 1 --max_num_jobs 50000 --job_max_runtime 600 --in_bucket pictureweb --out_bucket picturewebsolve --type "rbf" --gamma 1e3 --linear_kernel_key "XXT(X_train_fisher_vector_sharded)"


echo "Computing test kernel"
python compute_kernel.py "X_train_fisher_vector_sharded" --pywren_mode lambda --tasks_per_job 1 --max_num_jobs 50000 --job_max_runtime 600 --in_bucket pictureweb --out_bucket picturewebsolve --type "rbf" --gamma 1e3 --linear_kernel_key "XYT(X_train_fisher_vector_sharded, X_test_fisher_vector_sharded)" --test_key "X_test_fisher_vector_sharded"






