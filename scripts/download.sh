mkdir -p ./checkpoints

wget https://huggingface.co/spaces/xinyu1205/recognize-anything/resolve/main/ram_swin_large_14m.pth -O ./checkpoints/ram_swin_large_14m.pth
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth -O ./checkpoints/scaled_online.pth
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt -O ./checkpoints/sam2.1_hiera_large.pt
wget https://github.com/hkchengrex/Tracking-Anything-with-DEVA/releases/download/v1.0/DEVA-propagation.pth -O ./checkpoints/DEVA-propagation.pth
gdown --fuzzy https://drive.google.com/file/d/18d5M3nl3AxbG4ZkT7wssvMXZXbmXrnjz/view?usp=sharing -O ./checkpoints/ 
gdown --fuzzy https://drive.google.com/file/d/1S_T7DzqBXMtr0voRC_XUGn1VTnPk_7Rm/view?usp=sharing -O ./checkpoints/ 
