# Download the NYU train dataset used in the paper
mkdir dataset
mkdir dataset/nyu_depth_v2
echo Downloading dataset...
python ./utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP ./dataset/nyu_depth_v2/sync.zip
echo Unzipping dataset...
unzip -q ./dataset/nyu_depth_v2/sync.zip -d ./dataset/nyu_depth_v2/
