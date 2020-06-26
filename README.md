
# Note
This repository is a TensorFlow implementation of the BTS Depth Estimation model using tf.keras in Tensorflow 2, without using any custom C++ kernels/ops. It is forked from [original repository](https://github.com/cogaplex-bts/bts). Please submit an issue if you encounter incompatibilities with other minor versions of Tensorflow 2. \

***Currently, if you use Tensorflow 2.1.0 installed using pip, you may encounter a segmentation fault with certain GPU configurations. If that is the case, please install Tensorflow from source.***

# BTS
From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation   
[arXiv](https://arxiv.org/abs/1907.10326)  
[Supplementary material](https://arxiv.org/src/1907.10326v4/anc/bts_sm.pdf) 

## Video Demo 1
[![Screenshot](https://img.youtube.com/vi/2fPdZYzx9Cg/maxresdefault.jpg)](https://www.youtube.com/watch?v=2fPdZYzx9Cg)
## Video Demo 2
[![Screenshot](https://img.youtube.com/vi/1J-GSb0fROw/maxresdefault.jpg)](https://www.youtube.com/watch?v=1J-GSb0fROw)

## Evaluation with [NYU Depth V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)
```shell
$ cd ~/workspace/bts-fully-tf/
$ chmod +x *.sh
$ ./init_nyu_test_files.sh
$ mkdir models
# Get BTS model trained with NYU Depth V2
$ python utils/download_from_gdrive.py 1KgLFgZa9U4X7Kq1_U908idFo_GBfu04d models/bts_nyu.zip
$ unzip models/bts_nyu.zip -d ./models/
```
Once the preparation steps completed, you can evaluate BTS using following commands.
```
$ cd ~/workspace/bts-fully-tf/
$ python bts_test.py arguments_test_nyu.txt
```
You should see outputs like this:
```
Raw png files reading done
Evaluating 654 files
GT files reading done
0 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.880,   0.978,   0.996,   0.113,   0.060,   0.356,   0.142,  11.333,   0.048
Done.
```
A single RTX 2080 Ti takes about 34 seconds to process 654 testing images. \
A single TPU pod with 8 cores takes about 44 seconds to process 654 testing images.

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
$ python convert_data.py arguments_convert_nyu.txt
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
Run script "sync_project_frames_multi_threads.m" using MATLAB to get synchronized RGB and depth images.
This will save rgb-depth pairs in "~/workspace/bts-fully-tf/dataset/nyu_depth_v2/sync/".
If you want to train this model on a TPU, please convert the dataset to the tfrecord format and store it in a Google Cloud Storage Bucket as shown above.
Once the dataset is ready, you can train the network using the following command for GPU training.
```
$ cd ~/workspace/bts-fully-tf/
$ python bts_main.py arguments_train_nyu.txt
```
If you are training the network on a TPU or wish to use a Google Storage Bucket to store the training data, please use this command instead.
```
$ python bts_main.py arguments_train_nyu_gcloud.txt
```
You can check the training using tensorboard:
```
$ tensorboard --logdir ./models/bts_nyu/
```
Open localhost:6006 with your favorite browser to see the progress of training.

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
$ python convert_data.py arguments_convert_eigen.txt
$ gcloud config set project {project_id}
$ gsutil cp kitti-eigen-sync-compressed.tfrecord gs://{bucket_name}/
$ rm kitti-eigen-sync-compressed.tfrecord
```
Finally, we can train our network with the following command for GPU training.
```
$ cd ~/workspace/bts-fully-tf/
$ python bts_main.py arguments_train_eigen.txt
```
If you are training the network on a TPU or wish to use a Google Storage Bucket to store the training data, please use this command instead.
```
$ python bts_main.py arguments_train_eigen_gcloud.txt
```
## Testing and Evaluation with [KITTI](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)
Once you have KITTI dataset and official ground truth depthmaps, you can test and evaluate our model with following commands.
```
# Get KITTI model trained with KITTI Eigen split
$ cd ~/workspace/bts
$ python utils/download_from_gdrive.py 1w4WbSQxui8GTDEsjX5xb4m7_-5yCznhQ models/bts_eigen.zip
$ cd models && unzip bts_eigen.zip
```
Test and save results.
```
$ cd ~/workspace/bts
$ python bts_test.py arguments_test_eigen.txt
```
This will save results to ./result_bts_eigen.
Finally, we can evaluate the prediction results with
```
$ python eval_with_pngs.py --pred_path ./result_bts_eigen/raw/ --gt_path ../dataset/kitti_dataset/data_depth_annotated/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --do_kb_crop --garg_crop
```
You should see outputs like this:
```
GT files reading done
45 GT files missing
Computing errors
     d1,      d2,      d3,  AbsRel,   SqRel,    RMSE, RMSElog,   SILog,   log10
  0.951,   0.993,   0.998,   0.064,   0.256,   2.796,   0.100,   9.175,   0.028
Done.
```

## License
Copyright (C) 2019 Jin Han Lee, Myung-Kyu Han, Dong Wook Ko and Il Hong Suh \
Adapted to Tensorflow 2 with Keras by Clarence Chen.
This Software is licensed under GPL-3.0-or-later.
