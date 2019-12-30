cd /content/bts-fully-tf/utils
# Get official NYU Depth V2 split file
wget http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat
# Convert mat file to image files
python extract_official_train_test_set_from_mat.py nyu_depth_v2_labeled.mat splits.mat ../../dataset/nyu_depth_v2/official_splits/
cd ..
mkdir models
# Get BTS model trained with NYU Depth V2
python utils/download_from_gdrive.py 1KgLFgZa9U4X7Kq1_U908idFo_GBfu04d models/bts_nyu.zip
cd models
unzip bts_nyu.zip
cd /content/bts-fully-tf
python bts_test.py arguments_test_nyu.txt