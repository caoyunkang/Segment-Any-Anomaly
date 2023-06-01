import argparse

from tqdm import tqdm

import SAA as SegmentAnyAnomaly
from datasets import *
from utils.csv_utils import *
from utils.eval_utils import *
from utils.metrics import *
from utils.training_utils import *


def eval(
        # model-related
        model,
        train_data: DataLoader,
        test_data: DataLoader,

        # visual-related
        resolution,
        is_vis,

        # experimental parameters
        dataset,
        class_name,
        cal_pro,
        img_dir,
        k_shot,
        experiment_indx,
        device: str
):
    similarity_maps = []
    scores = []
    test_imgs = []
    gt_list = []
    gt_mask_list = []
    names = []

    for (data, mask, label, name, img_type) in tqdm(test_data):

        for d, n, l, m in zip(data, name, label, mask):
            d = d.numpy()
            l = l.numpy()
            m = m.numpy()
            m[m > 0] = 1

            test_imgs += [d]
            names += [n]
            gt_list += [l]
            gt_mask_list += [m]

            score, appendix = model(d)
            scores += [score]

            similarity_map = appendix['similarity_map']
            similarity_maps.append(similarity_map)

    test_imgs, scores, gt_mask_list = specify_resolution(
        test_imgs, scores, gt_mask_list,
        resolution=(resolution, resolution)
    )
    _, similarity_maps, _ = specify_resolution(
        test_imgs, similarity_maps, gt_mask_list,
        resolution=(resolution, resolution)
    )

    scores = normalize(scores)
    similarity_maps = normalize(similarity_maps)

    np_scores = np.array(scores)
    img_scores = np_scores.reshape(np_scores.shape[0], -1).max(axis=1)

    if dataset in ['visa_challenge']:
        save_results(img_scores, scores, f'{img_dir}/..', f'{k_shot}shot', f'{experiment_indx}', names,
                     use_defect_type=True)

    if dataset in ['visa_challenge']:
        result_dict = {'i_roc': 0, 'p_roc': 0, 'p_pro': 0,
                       'i_f1': 0, 'i_thresh': 0, 'p_f1': 0, 'p_thresh': 0,
                       'r_f1': 0}
    else:
        gt_list = np.stack(gt_list, axis=0)
        result_dict = metric_cal(np.array(scores), gt_list, gt_mask_list, cal_pro=cal_pro)

    if is_vis:
        plot_sample_cv2(
            names,
            test_imgs,
            {'SAA_plus': scores, 'Saliency': similarity_maps},
            gt_mask_list,
            save_folder=img_dir
        )

    return result_dict


def main(args):
    kwargs = vars(args)

    # prepare the experiment dir
    model_dir, img_dir, logger_dir, model_name, csv_path = get_dir_from_args(**kwargs)

    logger.info('==========running parameters=============')
    for k, v in kwargs.items():
        logger.info(f'{k}: {v}')
    logger.info('=========================================')

    # give some random seeds
    seeds = [111, 333, 999, 1111, 3333, 9999]
    kwargs['seed'] = seeds[kwargs['experiment_indx']]
    setup_seed(kwargs['seed'])

    if kwargs['use_cpu'] == 0:
        device = f"cuda:0"
    else:
        device = f"cpu"

    kwargs['device'] = device

    # get the train dataloader
    if kwargs['k_shot'] > 0:
        train_dataloader, train_dataset_inst = get_dataloader_from_args(phase='train', perturbed=False, **kwargs)
    else:
        train_dataloader, train_dataset_inst = None, None

    # get the test dataloader
    test_dataloader, test_dataset_inst = get_dataloader_from_args(phase='test', perturbed=False, **kwargs)

    # get the model
    model = SegmentAnyAnomaly.Model(
        dino_config_file=kwargs['dino_config_file'],
        dino_checkpoint=kwargs['dino_checkpoint'],
        sam_checkpoint=kwargs['sam_checkpoint'],
        box_threshold=kwargs['box_threshold'],
        text_threshold=kwargs['text_threshold'],
        out_size=kwargs['eval_resolution'],
        device=kwargs['device'],
    )

    general_prompts = SegmentAnyAnomaly.build_general_prompts(kwargs['class_name'])
    manual_promts = SegmentAnyAnomaly.manul_prompts[kwargs['dataset']][kwargs['class_name']]

    textual_prompts = general_prompts + manual_promts

    model.set_ensemble_text_prompts(textual_prompts, verbose=False)

    property_text_prompts = SegmentAnyAnomaly.property_prompts[kwargs['dataset']][kwargs['class_name']]
    model.set_property_text_prompts(property_text_prompts, verbose=False)

    model = model.to(device)

    metrics = eval(
        # model-related parameters
        model=model,
        train_data=train_dataloader,
        test_data=test_dataloader,

        # visual-related parameters
        resolution=kwargs['eval_resolution'],
        is_vis=True,

        # experimental parameters
        dataset=kwargs['dataset'],
        class_name=kwargs['class_name'],
        cal_pro=kwargs['cal_pro'],
        img_dir=img_dir,
        k_shot=kwargs['k_shot'],
        experiment_indx=kwargs['experiment_indx'],
        device=device
    )

    logger.info(f"\n")

    for k, v in metrics.items():
        logger.info(f"{kwargs['class_name']}======={k}: {v:.2f}")

    save_metric(metrics, dataset_classes[kwargs['dataset']], kwargs['class_name'],
                kwargs['dataset'], csv_path)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_args():
    parser = argparse.ArgumentParser(description='Anomaly detection')
    # data related parameters
    parser.add_argument('--dataset', type=str, default='mvtec',
                        choices=['mvtec', 'visa_challenge', 'visa_public', 'ksdd2', 'mtd'])
    parser.add_argument('--class-name', type=str, default='metal_nut')
    parser.add_argument('--k-shot', type=int, default=0) # no effect... just set it to 0.

    # experiment related parameters
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--vis', type=str2bool, choices=[True, False], default=True)
    parser.add_argument("--root-dir", type=str, default="./result")
    parser.add_argument("--cal-pro", type=str2bool, default=False)
    parser.add_argument("--experiment_indx", type=int, default=0) # no effect... just set it to 0.
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--use-cpu", type=int, default=0)

    # method related parameters
    parser.add_argument('--eval-resolution', type=int, default=400)
    parser.add_argument("--dino_config_file", type=str,
                        default='GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
                        help="path to config file")
    parser.add_argument(
        "--dino_checkpoint", type=str, default='weights/groundingdino_swint_ogc.pth', help="path to checkpoint file"
    )
    parser.add_argument(
        "--sam_checkpoint", type=str, default='weights/sam_vit_h_4b8939.pth', help="path to checkpoint file"
    )

    parser.add_argument("--box_threshold", type=float, default=0.1, help="box threshold")
    parser.add_argument("--text_threshold", type=float, default=0.1, help="text threshold")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    import os

    args = get_args()
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu_id}"
    main(args)
