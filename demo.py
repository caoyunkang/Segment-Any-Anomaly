import matplotlib.pyplot as plt
import SAA as SegmentAnyAnomaly
from utils.training_utils import *

if __name__ == '__main__':
    import os

    gpu_id = 0

    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{gpu_id}"


    dino_config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py'
    dino_checkpoint = 'weights/groundingdino_swint_ogc.pth'
    sam_checkpoint = 'weights/sam_vit_h_4b8939.pth'
    box_threshold = 0.1
    text_threshold = 0.1
    eval_resolution = 1024
    device = f"cuda:0"
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

    textual_prompts = ['color defect. hole. black defect. wick hole. spot. ', 'candle'] # detect prompts, filtered phrase
    property_text_prompts = 'the image of candle have 4 similar candle, with a maximum of 1 anomaly. The anomaly would not exceed 0.3 object area. '

    model.set_ensemble_text_prompts(textual_prompts, verbose=False)
    model.set_property_text_prompts(property_text_prompts, verbose=False)

    model = model.to(device)

    image_path = 'assets/candle.JPG'
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    score, appendix = model(image)

    similarity_map = appendix['similarity_map']

    image_show = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_show = cv2.resize(image_show, (eval_resolution, eval_resolution))
    similarity_map = cv2.resize(similarity_map, (eval_resolution, eval_resolution))
    score = cv2.resize(score, (eval_resolution, eval_resolution))

    plt.subplot(121)
    plt.imshow(image_show)
    plt.imshow(score, alpha=0.4,cmap='jet')
    plt.title('Anomaly Score')

    plt.subplot(122)
    plt.imshow(image_show)
    plt.imshow(similarity_map, alpha=0.4, cmap='jet')
    plt.title('Saliency')
    plt.show()
    # plt.savefig(os.path.join(root_dir, 'result_image.png'))

