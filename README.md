
# Note
This repository is a TensorFlow implementation of the BTS Depth Estimation model using tf.keras in Tensorflow 2, without using any custom C++ kernels/ops. It is forked from [original repository](https://github.com/cogaplex-bts/bts). Please submit an issue if you encounter incompatibilities with other minor versions of Tensorflow 2.

***Currently, if you use Tensorflow 2.1.0 installed using pip, you may encounter a segmentation fault with certain GPU configurations. If that is the case, please install Tensorflow from source.***

# TODO

## High Priority
- [x] Direct prediction of depth maps to saved PNG images
- [ ] Web server with live demo of results using Flask or node.js

## Medium Priority
- [x] Additional image classification backbone models using `tf.keras.applications`
- [ ] Random image rotation in the data preprocessing pipeline when training on TPU
- [ ] Validation split and hyperparameter tuning using optuna
- [ ] Depth map output in Tensorboard using validation split above

## Low priority
 - Support parallelized tfrecord sharding in `bts_convert_data.py`

# BTS
From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   
[arXiv](https://arxiv.org/abs/1907.10326)  
[Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf) 

## Demo Input Image/Output Depthmap Pairs
### Outdoor Scenes from KITTI Dataset
![RGB Photo](https://raw.githubusercontent.com/clarencechen/bts-fully-tf/master/demo/0000000140.png)
![Depth Map](https://raw.githubusercontent.com/clarencechen/bts-fully-tf/master/demo/2011_09_26_drive_0013_sync_0000000140.png)
![RGB Photo](https://raw.githubusercontent.com/clarencechen/bts-fully-tf/master/demo/0000000000.png)
![Depth Map](https://raw.githubusercontent.com/clarencechen/bts-fully-tf/master/demo/2011_09_26_drive_0046_sync_0000000000.png)
### Simulated Indoor Scenes (OOD Inputs with Model Trained on NYUv2 Dataset)
![RGB Photo](https://raw.githubusercontent.com/clarencechen/bts-fully-tf/master/demo/indoor1.png)
![Depth Map](https://raw.githubusercontent.com/clarencechen/bts-fully-tf/master/demo/default_indoor1.png)
## Evaluation with [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
```shell
$ cd ~/workspace/bts-fully-tf/
$ chmod +x *.sh
$ ./init_nyu_test_files.sh
```
Once the preparation steps completed, you can evaluate BTS using following commands.
```
$ cd ~/workspace/bts-fully-tf/
$ mkdir ./models/; mkdir ./models/bts_nyu/
$ gsutil -m cp -r gs://bts-tf2-model/bts_nyu/* ./models/bts_nyu/
$ python bts_eval.py args/test_nyu.txt
```
You should see outputs like this:
```
Now testing 654 images.
81/81 [==============================] - 44s 546ms/step
  silog, abs_rel,   log10,     rms,  sq_rel, log_rms,      d1,      d2,      d3
12.6411,  0.1205,   0.052,   0.432,   0.076,   0.157,   0.854,   0.972,   0.994
```
Note that the results shown above have been produced by a TPU-trained model with batch size 32 (4 per TPU core). Additional hyperparameter tuning and the addition of currently unsupported rotation augmentation may improve results further.

### Estimated Evaluation Time on Different Accelerators
A single RTX 2080 Ti takes about 41 seconds to process 654 testing images. \
A single Tesla T4 takes about 65 seconds to process 654 testing images. \
A single TPU pod with 8 cores takes about 44 seconds to process 654 testing images.

## Testing and Evaluation with [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
Once you have KITTI dataset and official ground truth depthmaps, you can test and evaluate our model with following commands.
```
$ cd ~/workspace/bts
$ mkdir ./models/; mkdir ./models/bts_eigen/
$ gsutil -m cp -r gs://bts-tf2-model/bts_eigen/* ./models/bts_eigen/
$ python bts_eval.py args/test_eigen.txt
```
You should see outputs like this:
```
Now testing 652 images.
81/81 [==============================] - 57s 702ms/step
  silog, abs_rel,   log10,     rms,  sq_rel, log_rms,      d1,      d2,      d3
10.2789,  0.0724,   0.032,   3.158,   0.320,   0.113,   0.937,   0.989,   0.998
```
Note that the results shown above have been produced by a TPU-trained model with batch size 32 (4 per TPU core). Additional hyperparameter tuning and the addition of currently unsupported rotation augmentation may improve results further.

## Preparation for Training
### NYU Depth V2
First, you need to download DenseNet-161 model pretrained with ImageNet.
```
$ cd ~/workspace/bts-fully-tf/
$ chmod +x *.sh
$ ./init_densenet_161.sh
```
Then, download the prepared dataset from the paper authors' Google Drive account.
```
$ ./init_nyu_train_files.sh
```
If you want to train this model on a TPU, you must use a Google Cloud Storage Bucket to store this dataset in tfrecord format due to data pipeline speed constraints for TPU training.
```
$ cd ~/workspace/bts-fully-tf/
$ python convert_data.py args/convert_nyu.txt
$ gcloud config set project {project_id}
$ gsutil cp nyu-depth-v2-compressed.tfrecord gs://{bucket_name}/
$ rm nyu-depth-v2-compressed.tfrecord
```
If you are unable obtain the dataset in this manner to due to Google Drive usage limits, you can prepare the dataset by yourself using original files from official site [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html).
There are two options for downloading original files: Single file downloading and Segmented-files downloading.

Single file downloading:
```
$ cd ~/workspace/bts-fully-tf/
$ ./init_nyu_data_raw.sh
```
Segmented-files downloading:
```
$ cd ~/workspace/dataset/nyu_depth_v2
$ ./init_nyu_data_raw_multiple.sh
```
Get official toolbox for rgb and depth synchronization.
```
$ cd ~/workspace/bts-fully-tf/utils/
$ wget http://cs.nyu.edu/~silberman/code/toolbox_nyu_depth_v2.zip
$ unzip toolbox_nyu_depth_v2.zip
$ cd toolbox_nyu_depth_v2
$ mv ../sync_project_frames_multi_threads.m .
$ mv ../train_scenes.txt .
```
Run the script `./sync_project_frames_multi_threads.m` using MATLAB to get synchronized RGB and depth images.
This will save rgb-depth pairs in `~/workspace/bts-fully-tf/dataset/nyu_depth_v2/sync/`.
If you want to train this model on a TPU, please convert the dataset to the tfrecord format and store it in a Google Cloud Storage Bucket as shown above.
Once the dataset is ready, you can train the network using the following command for GPU training.
```
$ cd ~/workspace/bts-fully-tf/
$ python bts_train.py args/train_nyu.txt --log_directory ./models/
```
If you are training the network on a TPU or wish to use a Google Storage Bucket to store the training data, please use this command instead.
```
$ python bts_train.py args/train_nyu_gcloud.txt --log_directory gs://{bucket_name}/
```
You can check the training using Tensorboard with logs either stored in a local directory:
```
$ tensorboard --logdir ./models/bts_nyu/tensorboard/
```
or stored in a Google Cloud Bucket:
```
$ tensorboard --logdir gs://{bucket_name}/bts_nyu/tensorboard/
```
Open `localhost:6006` with your favorite browser to see the progress of training.

### KITTI
You can also train BTS with KITTI dataset by following procedures.
First, download the ground truth depthmaps from [KITTI](http://www.cvlibs.net/download.php?file=data_depth_annotated.zip).
Then, download and unzip the raw dataset using following commands.
```
$ cd ~/workspace/bts-fully-tf/
$ ./init_kitti_files.sh
```
If you want to train this model on a TPU, you must use a Google Cloud Storage Bucket to store this dataset in tfrecord format due to data pipeline speed constraints for TPU training.
```
$ cd ~/workspace/bts-fully-tf/
$ python convert_data.py args/convert_eigen.txt
$ gcloud config set project {project_id}
$ gsutil -m cp kitti-eigen-sync-compressed.tfrecord_* gs://{bucket_name}/
$ rm kitti-eigen-sync-compressed.tfrecord
```
Finally, we can train our network with the following command for GPU training.
```
$ cd ~/workspace/bts-fully-tf/
$ python bts_train.py args/train_eigen.txt --log_directory ./models/
```
If you are training the network on a TPU or wish to use a Google Storage Bucket to store the training data, please use this command instead.
```
$ python bts_train.py args/train_eigen_gcloud.txt --log_directory gs://{bucket_name}/
```

## License
Copyright (C) 2019 Jin Han Lee, Myung-Kyu Han, Dong Wook Ko and Il Hong Suh \
Adapted to Tensorflow 2 with Keras by Clarence Chen.
This Software is licensed under GPL-3.0-or-later.
