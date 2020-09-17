# CNN for Strabismus Detection

**Notice**: as the project is still ongoing, the currently provided models are not well trained yet!

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


