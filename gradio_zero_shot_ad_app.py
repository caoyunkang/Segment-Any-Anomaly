import gradio as gr
import argparse
import os

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# segment anything
from SAM.segment_anything import build_sam, SamPredictor 
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

def load_image(image_pil):
    # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold, area_threshold, category, with_logits=True, device="cpu"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    boxes_area = boxes_filt[:, 2] * boxes_filt[:, 3]
    filt_mask = torch.bitwise_and((logits_filt.max(dim=1)[0] > box_threshold), (boxes_area < area_threshold))

    if torch.sum(filt_mask) == 0: # in case there are no matches
        filt_mask = torch.argmax(logits_filt.max(dim=1)[0])
        logits_filt = logits_filt[filt_mask].unsqueeze(0)  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask].unsqueeze(0)
    else:
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)

    # build pred
    pred_phrases = []
    boxes_filt_category = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if pred_phrase.count(category) > 0: # we don't want to predict the category
            continue

        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)
        boxes_filt_category.append(box)
    boxes_filt_category = torch.stack(boxes_filt_category, dim=0)

    return boxes_filt_category, pred_phrases


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2)) 
    ax.text(x0, y0, label)


def run_grounded_sam(image_pil, text_prompt, task_type, box_threshold, text_threshold, area_threshold, category):
    assert text_prompt, 'text_prompt is not found!'

    # load image
    image_pil, image = load_image(image_pil.convert("RGB"))
    # load dino model
    model = load_model(config_file, grounded_checkpoint, device=device)
    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        model, image, text_prompt, box_threshold, text_threshold, area_threshold, category=category, device=device
    )

    size = image_pil.size

    if task_type == 'seg' or task_type == 'inpainting':
        # initialize SAM
        predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint))
        image = np.array(image_pil)
        predictor.set_image(image)

        H, W = size[1], size[0]
        for i in range(boxes_filt.size(0)):
            boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
            boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
            boxes_filt[i][2:] += boxes_filt[i][:2]

        boxes_filt = boxes_filt.cpu()
        transformed_boxes = predictor.transform.apply_boxes_torch(boxes_filt, image.shape[:2])

        masks, _, _ = predictor.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = transformed_boxes,
            multimask_output = False,
        )

        # masks: [1, 1, 512, 512]

    if task_type == 'det':
        pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H, W
            "labels": pred_phrases,
        }
        image_pil = plot_boxes_to_image(image_pil, pred_dict)[0]
        return image_pil

    elif task_type == 'seg':
        assert sam_checkpoint, 'sam_checkpoint is not found!'

        # draw output image
        fig = plt.figure(figsize=(size[0] / 100, size[1] / 100))
        plt.imshow(image)
        for mask in masks:
            show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
        for box, label in zip(boxes_filt, pred_phrases):
            show_box(box.numpy(), plt.gca(), label)
        plt.axis('off')

        # move the white margin
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)

        # plt.figure -> pil
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        image_pil = Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb())
        return image_pil


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser("Grounded SAM demo", add_help=True)
    parser.add_argument("--debug", action="store_true", help="using debug mode")
    parser.add_argument("--share", action="store_true", help="share the app")
    args = parser.parse_args()

    config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    grounded_checkpoint = "weights/groundingdino_swint_ogc.pth"
    sam_checkpoint = 'weights/sam_vit_h_4b8939.pth' 
    output_dir = "outputs"
    device ="cuda"

    block = gr.Blocks().queue()
    with block:
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(source='upload', type="pil")
                text_prompt = gr.Textbox(label="Detection Prompt")
                task_type = gr.Textbox(label="task type: det/seg")
                category_type = gr.Textbox(label="Background: to be filtered")
                run_button = gr.Button(label="Run")
                with gr.Accordion("Advanced options", open=True):
                    box_threshold = gr.Slider(
                        label="Box Threshold", minimum=0.0, maximum=1.0, value=0.3, step=0.001
                    )
                    text_threshold = gr.Slider(
                        label="Text Threshold", minimum=0.0, maximum=1.0, value=0.25, step=0.001
                    )
                    area_threshold = gr.Slider(
                        label="Area Threshold", minimum=0.0, maximum=1.0, value=0.9, step=0.001
                    )

            with gr.Column():
                gallery = gr.Image(
                    type="pil",
                    label="Result",
                    tool="select"
                ).style(full_width=True, full_height=True)

        run_button.click(fn=run_grounded_sam, inputs=[
                        input_image, text_prompt, task_type, box_threshold, text_threshold, area_threshold, category_type], outputs=[gallery])


        block.launch(server_name='0.0.0.0', server_port=8000, debug=args.debug, share=args.share)
