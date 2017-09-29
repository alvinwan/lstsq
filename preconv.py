from featurization.conv import output_shape


print(output_shape((1, 210, 160, 3), batch_feature_size=64, num_feature_batches=8,
    data_batch_size=100, patch_size=10, pool_size=150))