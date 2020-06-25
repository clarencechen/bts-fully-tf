# Download the NYU raw dataset used in the paper
mkdir dataset
mkdir dataset/nyu_depth_v2
mkdir dataset/nyu_depth_v2/raw
echo Downloading dataset...
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_raw.zip -P ./dataset/nyu_depth_v2/raw/
echo Unzipping dataset...
unzip -q ./dataset/nyu_depth_v2/raw/nyu_depth_v2_raw.zip -d ./dataset/nyu_depth_v2/raw/
