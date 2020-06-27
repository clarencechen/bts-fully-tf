rm -r sample_data/
mkdir dataset
mkdir dataset/kitti
# Install dependencies for parallel downloading and unzipping
apt-get install dos2unix parallel
# Download the official KITTI depth maps used in the paper
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip
unzip -q data_depth_annotated.zip -d ./dataset/kitti/
# Remove original zip file and unused images to save >20GB disk space
rm data_depth_annotated.zip
rm -rf dataset/kitti/*/*/*/*/image_03/
# Strip unnecessary line feeds from KITTI image archive list
dos2unix kitti_archives_to_download.txt
# Only unzip used image types and delete orginal zip files to save >100GB disk space
cat kitti_archives_to_download.txt | parallel --compress "wget -nv {} && unzip -q {/} '*/{/.}/image_02/*' -d ./dataset/kitti/ && rm {/}"
# Unzip calibration files which don't have 'image_02' directories in the archive
ls *calib.zip | parallel 'unzip {/} -d ./dataset/kitti/ && rm {/}'
# Merge KITTI ground truth depth maps together into one directory
mkdir dataset/kitti/data_depth_annotated/
mv ./dataset/kitti/train/* ./dataset/kitti/val/* ./dataset/kitti/data_depth_annotated/
