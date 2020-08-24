# CNN for Strabismus Detection

Sample commands for detecting strabismus

```

$ python CNN4Strabismus.py -m model_03-31-20.h5 -i data/test/patient/1609_r.jpg
Using TensorFlow backend.
2020-08-23 22:00:05.629362: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2020-08-23 22:00:05.637607: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
dim: (100, 400)
dim after resize: (100, 350)
image.shape: (100, 350, 3)
the subject is diagnosed as strabismus!

$ python CNN4Strabismus.py -m model_03-31-20.h5 -i data/test/healthy/806.jpg
Using TensorFlow backend.
2020-08-23 22:01:07.359655: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2020-08-23 22:01:07.369629: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
dim: (100, 447)
dim after resize: (100, 350)
image.shape: (100, 350, 3)
the subject is diagnosed as healthy!

```

```

$ python CNN4Strabismus.py -m model_03-31-20.h5 -i data/raw/patient/eso_008.jpg --raw
Using TensorFlow backend.
2020-08-23 22:03:50.957813: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2020-08-23 22:03:50.967689: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
dim: (83, 257)
dim after resize: (100, 350)
image.shape: (100, 350, 3)
the subject is diagnosed as strabismus!


$ python CNN4Strabismus.py -m model_03-31-20.h5 -i data/raw/healthy/healthy_001.png --raw
Using TensorFlow backend.
2020-08-23 22:01:52.710607: I tensorflow/core/platform/cpu_feature_guard.cc:141] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX AVX2
2020-08-23 22:01:52.719969: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 8. Tune using inter_op_parallelism_threads for best performance.
dim: (334, 1212)
dim after resize: (100, 350)
image.shape: (100, 350, 3)
the subject is diagnosed as healthy!

```

