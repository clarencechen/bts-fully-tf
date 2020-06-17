# Get DenseNet-161 model pretrained with ImageNet
mkdir models
mkdir models/densenet161_imagenet
echo Downloading pretrained model...
python ./utils/download_from_gdrive.py 0Byy2AcGyEVxfUDZwVjU2cFNidTA models/densenet161_imagenet/weights.h5