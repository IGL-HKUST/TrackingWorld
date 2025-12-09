pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# -------------------------- For RAM+GPT -------------------------
pip install git+https://github.com/xinyu1205/recognize-anything.git
pip install dotenv
pip install transformers
pip install timm
pip install fairscale
pip install openai

# -------------------------- For DEVA -------------------------
cd submodules/Tracking-Anything-with-DEVA
pip install -e .
cd ..

# -------------------------- For Grounded-SAM-2 -------------------------
cd submodules/Grounded-SAM-2
pip install -e .
pip install --no-build-isolation -e grounding_dino
cd ..

# -------------------------- For Unidepth-------------------------
cd submodules/UniDepth
pip install . --no-deps
pip install einops
pip install wandb
pip install xformers==0.0.29
wget https://raw.githubusercontent.com/AbdBarho/xformers-wheels/refs/heads/main/xformers/components/attention/nystrom.py -O ./unidepth/layers/nystrom.py  # To ensure compatibility of pytorch 2.5.1
sed -i 's/from xformers\.components\.attention import NystromAttention/from .nystrom import NystromAttention/g' unidepth/layers/nystrom_attention.py
cd ..

# -------------------------- For TrackingWorld -------------------------

pip install imageio
pip install configargparse
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# -------------------------- For Evaluation -------------------------
pip install evo

# -------------------------- For Visualization-------------------------
pip install rerun-sdk
pip install viser