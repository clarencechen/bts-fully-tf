# Create requisite directories
mkdir models
mkdir dataset
mkdir models/densenet161_imagenet
mkdir dataset/nyu_depth_v2
# Get DenseNet-161 model pretrained with ImageNet
echo Downloading pretrained model...
python ./utils/download_from_gdrive.py 0Byy2AcGyEVxfUDZwVjU2cFNidTA models/densenet161_imagenet/weights.h5
python bts_main.py arguments_train_nyu_tpu.txt
