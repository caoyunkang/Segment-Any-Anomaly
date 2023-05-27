import glob
import os
import random

visa_public_classes = ['candle', 'capsules', 'cashew', 'chewinggum',
                       'fryum', 'macaroni1', 'macaroni2',
                       'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

VISA_DIR = '../datasets/VisA_pytorch/1cls'


def load_visa_public(category, k_shot, experiment_indx):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.JPG")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.JPG")
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '.png') for s in
                            img_paths]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in visa_public_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    test_img_path = os.path.join(VISA_DIR, category, 'test')
    train_img_path = os.path.join(VISA_DIR, category, 'train')
    ground_truth_path = os.path.join(VISA_DIR, category, 'ground_truth')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, ground_truth_path)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    selected_train_img_tot_paths = []
    selected_train_gt_tot_paths = []
    selected_train_tot_labels = []
    selected_train_tot_types = []

    if k_shot > 0:
        full_index = range(len(train_img_tot_paths))

        selected_index = random.sample(full_index, k_shot)
        selected_train_img_tot_paths = [train_img_tot_paths[k] for k in selected_index]
        selected_train_gt_tot_paths = [train_gt_tot_paths[k] for k in selected_index]
        selected_train_tot_labels = [train_tot_labels[k] for k in selected_index]
        selected_train_tot_types = [train_tot_types[k] for k in selected_index]

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels,
            selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
