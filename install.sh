# PyTorch
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# SAM and DINO
python -m pip install -e segment_anything
python -m pip install -e GroundingDINO
pip install --upgrade diffusers[torch]
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel
pip install transformers
pip install addict
pip install yapf
pip install timm

# UI
pip install gradio

# weights
cd GroundedSAM-zero-shot-anomaly-detection
mkdir weights
cd ./weights/
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
