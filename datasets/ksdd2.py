import os

import cv2

ksdd2_classes = ['ksdd2']

KSDD2_DIR = '../datasets/KolektorSDD2'


def load_ksdd2(category, k_shot, experiment_indx):
    def load_phase(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        image_paths = os.listdir(root_path)
        image_names = [f[:5] for f in image_paths]

        image_names = sorted(list(set(image_names)))

        for name in image_names:
            img_path = os.path.join(root_path, f'{name}.png')
            gt_path = os.path.join(root_path, f'{name}_GT.png')

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt.sum() > 0:
                label = 1
                img_type = 'bad'
            else:
                label = 0
                img_type = 'good'
            img_tot_paths.append(img_path)
            gt_tot_paths.append(gt_path)
            tot_labels.append(label)
            tot_types.append(img_type)

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in ksdd2_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    test_img_path = os.path.join(KSDD2_DIR, 'train')
    train_img_path = os.path.join(KSDD2_DIR, 'test')

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path)

    selected_train_img_tot_paths = []
    selected_train_gt_tot_paths = []
    selected_train_tot_labels = []
    selected_train_tot_types = []

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels,
            selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
