from featurization.conv import output_shape

shape = (1, 160, 160, 3)
patch_sizes = tuple(range(2, 20, 2))

shapes = []
for patch_size in patch_sizes:
  pool_size = 50
  result = (0, 0, 100, 100)
  while result[-2:] != (2, 2):
    result = output_shape((1, 160, 160, 3), batch_feature_size=64, num_feature_batches=8,
        data_batch_size=100, patch_size=patch_size, pool_size=pool_size)
    pool_size += 1
  shapes.append(result)
  print('shape:', shape, '// patch size:', patch_size, '// pool size:', pool_size)