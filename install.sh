# create new conda env
conda create -n SAA python=3.9
source activate SAA

# PyTorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# $ProjectRoot: the root you save our project, e.g., /home/anyad/VAND-solution
ProjectRoot=/home/anyad/VAND-solution
cd $ProjectRoot

# SAM and DINO
cd ./GroundingDINO
pip install -e .
cd ../SAM
pip install -e .

pip install setuptools==59.5.0
pip install --upgrade diffusers[torch]
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
pip install transformers
pip install addict
pip install yapf
pip install timm
pip install loguru
pip install tqdm
pip install scikit-image
pip install scikit-learn
pip install pandas
pip install tensorboard
pip install seaborn
pip install open_clip_torch
pip install SciencePlots
pip install timm
pip install einops
pip install gradio

# weights
cd ../
mkdir weights
cd ./weights/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth


