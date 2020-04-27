# Collaborating on FL & crypto benchmarks

In order for the final research product to include valuable
benchmarks, we should use a shared common lib for tasks that
are necessary for both PySyft & TFF/TFE stacks.

#### What are good candidates for this common lib?
- dataset processing/labeling code (e.g. numpy/pandas)
- client allocation code (should be python/numpy/pandas)
- splitting clients into training/test

This enables some level of benchmark reproducibility across
TF/PySyft experiments.

#### What are not good candidates for this common lib?
- data loading (e.g. torch.data.DataLoader, tf.data.Dataset)
- model training/testing code
- model inference code
- code with heavy dependencies
