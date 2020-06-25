# Download the NYU raw dataset used in the paper
mkdir dataset
mkdir dataset/nyu_depth_v2
mkdir dataset/nyu_depth_v2/raw && cd dataset/nyu_depth_v2/raw
# Install prerequisites for parallel downloading and unzipping if necessary
apt-get install parallel aria2
aria2c -x 16 -i ../../../utils/nyudepthv2_archives_to_download.txt
cd ~/workspace/bts-fully-tf/
python utils/download_from_gdrive.py 1xBwO6qU8UCS69POJJ0-9luaG_1pS1khW ./dataset/nyu_depth_v2/raw/bathroom_0039.zip
python utils/download_from_gdrive.py 1IFoci9kns6vOV833S7osV6c5HmGxZsBp ./dataset/nyu_depth_v2/raw/bedroom_0076a.zip
python utils/download_from_gdrive.py 1ysSeyiOiOI1EKr1yhmKy4jcYiXdgLP4f ./dataset/nyu_depth_v2/raw/living_room_0018.zip
python utils/download_from_gdrive.py 1QkHkK46VuKBPszB-mb6ysFp7VO92UgfB ./dataset/nyu_depth_v2/raw/living_room_0019.zip
python utils/download_from_gdrive.py 1g1Xc3urlI_nIcgWk8I-UaFXJHiKGzK6w ./dataset/nyu_depth_v2/raw/living_room_0020.zip
cd dataset/nyu_depth_v2/raw/ && parallel unzip ::: *.zip
cd ~/workspace/bts-fully-tf/
