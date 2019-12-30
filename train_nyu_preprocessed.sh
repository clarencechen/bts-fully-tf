# Get DenseNet-161 model pretrained with ImageNet
cd /content/bts-fully-tf
mkdir models
python utils/download_from_gdrive.py 1rn7xBF5eSISFKL2bIa8o3d8dNnsrlWfJ models/densenet161_imagenet.zip
cd models && unzip densenet161_imagenet.zip
# Download the NYU dataset used in the paper
cd /content/bts-fully-tf
python utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP ../dataset/nyu_depth_v2/sync.zip
cd ../dataset/nyu_depth_v2
unzip sync.zip
cd /content/bts-fully-tf
python bts_main.py arguments_train_nyu.txt