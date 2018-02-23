#!/bin/bash

echo "Computing test kernel"
python compute_kernel.py "X_train_scattering_J_4" --gamma 0.01 --pywren_mode standalone --tasks_per_job 40 --max_num_jobs 50000 --job_max_runtime 9000 --test_key "X_test_scattering_J_4" --in_bucket vaishaalpywrenlinalg --out_bucket picturewebsolve --type "liner" --local

#echo "Computing train kernel"
#python compute_kernel.py "X_train_scattering_J_4" --gamma 0.01 --pywren_mode standalone --tasks_per_job 100 --max_num_jobs 50000 --job_max_runtime 9000 --in_bucket vaishaalpywrenlinalg --out_bucket picturewebsolve --type "linear"






