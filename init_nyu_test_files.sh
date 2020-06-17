# Get official NYU Depth V2 split file
mkdir dataset
mkdir dataset/nyu_depth_v2
echo Downloading dataset...
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat -P ./dataset/
# Convert mat file to image files
python ./utils/extract_official_train_test_set_from_mat.py ./dataset/nyu_depth_v2_labeled.mat ./utils/splits.mat ./dataset/nyu_depth_v2/official_splits/
