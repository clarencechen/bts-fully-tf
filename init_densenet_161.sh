# Get DenseNet-161 model pretrained with ImageNet
mkdir models
mkdir models/densenet161_imagenet
echo Downloading pretrained model...
gsutil cp gs://densenet161-imagenet/weights.h5 ./models/densenet161_imagenet/