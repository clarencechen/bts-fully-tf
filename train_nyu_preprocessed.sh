# Get DenseNet-161 model pretrained with ImageNet
cd /content/bts-fully-tf
python utils/download_from_gdrive.py 0Byy2AcGyEVxfUDZwVjU2cFNidTA models/densenet161_imagenet/weights.h5
# Download the NYU dataset used in the paper
cd /content/bts-fully-tf
python utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP dataset/nyu_depth_v2/sync.zip
cd dataset/nyu_depth_v2
unzip -q sync.zip
cd /content/bts-fully-tf
python bts_main.py arguments_train_nyu.txt