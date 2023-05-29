import os
import sys

os.chdir('GroundingDINO/')
os.system('pip install -e .')
os.chdir('../SAM')
os.system('pip install -e .')
os.system('pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel gradio loguru transformers timm addict yapf loguru tqdm scikit-image scikit-learn pandas tensorboard seaborn open_clip_torch  einops')
os.system('pip install torch==1.10.0 torchvision==0.11.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html')

os.chdir('..')
os.mkdir('weights')
os.chdir('./weights')
os.system('wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth')
os.system('wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth')
os.chdir('..')

import sys
sys.path.append('./GroundingDINO')
sys.path.append('./SAM')
sys.path.append('.')
import matplotlib.pyplot as plt
import SAA as SegmentAnyAnomaly
from utils.training_utils import *
import os



dino_config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
dino_checkpoint = 'weights/groundingdino_swint_ogc.pth'
sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'
box_threshold = 0.1
text_threshold = 0.1
eval_resolution = 1024
device = f"cpu"
root_dir = 'result'

# get the model
model = SegmentAnyAnomaly.Model(
    dino_config_file=dino_config_file,
    dino_checkpoint=dino_checkpoint,
    sam_checkpoint=sam_checkpoint,
    box_threshold=box_threshold,
    text_threshold=text_threshold,
    out_size=eval_resolution,
    device=device,
)

model = model.to(device)

import cv2
import numpy as np
import gradio as gr


def process_image(heatmap, image):
    heatmap = heatmap.astype(float)
    heatmap = (heatmap - heatmap.min()) / heatmap.max() * 255
    heatmap = heatmap.astype(np.uint8)
    heat_map = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    visz_map = cv2.addWeighted(heat_map, 0.5, image, 0.5, 0)
    visz_map = cv2.cvtColor(visz_map, cv2.COLOR_BGR2RGB)

    visz_map = visz_map.astype(float)
    visz_map = visz_map / visz_map.max()
    return visz_map


def func(image, anomaly_description, object_name, object_number, mask_number, area_threashold):
    textual_prompts = [
        [anomaly_description, object_name]
    ]  # detect prompts, filtered phrase
    property_text_prompts = f'the image of {object_name} have {object_number} dissimilar {object_name}, with a maximum of {mask_number} anomaly. The anomaly would not exceed {area_threashold} object area. '

    model.set_ensemble_text_prompts(textual_prompts, verbose=True)
    model.set_property_text_prompts(property_text_prompts, verbose=True)

    image = cv2.resize(image, (eval_resolution, eval_resolution))
    score, appendix = model(image)
    similarity_map = appendix['similarity_map']

    image_show = cv2.resize(image, (eval_resolution, eval_resolution))
    similarity_map = cv2.resize(similarity_map, (eval_resolution, eval_resolution))
    score = cv2.resize(score, (eval_resolution, eval_resolution))

    viz_score = process_image(score, image_show)
    viz_sim = process_image(similarity_map, image_show)

    return viz_score, viz_sim


with gr.Blocks() as demo:
    image = gr.Image(label="Image")
    anomaly_description = gr.Textbox(label="Anomaly Description")
    object_name = gr.Textbox(label="Object Name")
    object_number = gr.Textbox(label="Object Number")
    mask_number = gr.Textbox(label="Mask Number")
    area_threashold = gr.Textbox(label="Area Threshold")

    anomaly_score = gr.Image(label="Anomaly Score")
    saliency_map = gr.Image(label="Saliency Map")

    greet_btn = gr.Button("Inference")
    greet_btn.click(fn=func,
                    inputs=[image, anomaly_description, object_name, object_number, mask_number, area_threashold],
                    outputs=[anomaly_score, saliency_map], api_name="Segment-Any-Anomaly")

demo.launch()


