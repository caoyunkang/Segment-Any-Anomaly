import glob
import os

mvtec_classes = ['carpet', 'grid', 'leather', 'tile', 'wood',
                 'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut', 'pill',
                 'screw', 'toothbrush', 'transistor', 'zipper']

MVTEC2D_DIR = '../datasets/mvtec_anomaly_detection'


def load_mvtec(category, k_shot, experiment_indx):
    def load_phase(root_path, gt_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        defect_types = os.listdir(root_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(root_path, defect_type) + "/*.png")
                gt_paths = [os.path.join(gt_path, defect_type, os.path.basename(s)[:-4] + '_mask.png') for s in
                            img_paths]
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in mvtec_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    test_img_path = os.path.join(MVTEC2D_DIR, category, 'test')
    train_img_path = os.path.join(MVTEC2D_DIR, category, 'train')
    ground_truth_path = os.path.join(MVTEC2D_DIR, category, 'ground_truth')

    if k_shot == 0:
        training_indx = []
    else:
        seed_file = os.path.join('./datasets/seeds_mvtec', category, 'selected_samples_per_run.txt')
        with open(seed_file, 'r') as f:
            files = f.readlines()
        begin_str = f'{experiment_indx}-{k_shot}: '

        training_indx = []
        for line in files:
            if line.count(begin_str) > 0:
                strip_line = line[len(begin_str):-1]
                index = strip_line.split(' ')
                training_indx = index

    train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
    train_tot_types = load_phase(train_img_path, ground_truth_path)

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path, ground_truth_path)

    selected_train_img_tot_paths = []
    selected_train_gt_tot_paths = []
    selected_train_tot_labels = []
    selected_train_tot_types = []

    for img_path, gt_path, label, defect_type in zip(train_img_tot_paths, train_gt_tot_paths, train_tot_labels,
                                                     train_tot_types):
        if os.path.basename(img_path[:-4]) in training_indx:
            selected_train_img_tot_paths.append(img_path)
            selected_train_gt_tot_paths.append(gt_path)
            selected_train_tot_labels.append(label)
            selected_train_tot_types.append(defect_type)

    return (selected_train_img_tot_paths, selected_train_gt_tot_paths, selected_train_tot_labels,
            selected_train_tot_types), \
           (test_img_tot_paths, test_gt_tot_paths, test_tot_labels, test_tot_types)
