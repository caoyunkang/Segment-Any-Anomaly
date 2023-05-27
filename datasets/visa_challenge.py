import glob
import os

visa_challenge_classes = ['candle', 'capsules', 'cashew', 'chewinggum',
                          'fryum', 'macaroni1', 'macaroni2',
                          'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum']

VISA_ZERO_SHOT_DIR = '../datasets/vand_0shot'
VISA_FEW_SHOT_DIR = '../datasets/vand_kshot'


def load_visa_challenge(category, k_shot, experiment_indx):
    def load_phase(root_path):
        img_tot_paths = []
        gt_tot_paths = []
        tot_labels = []
        tot_types = []

        img_paths = glob.glob(os.path.join(root_path) + "/*.*")
        img_tot_paths.extend(img_paths)
        gt_tot_paths.extend([0] * len(img_paths))
        tot_labels.extend([0] * len(img_paths))
        tot_types.extend(['Unknown'] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    assert category in visa_challenge_classes
    assert k_shot in [0, 1, 5, 10]
    assert experiment_indx in [0, 1, 2]

    if k_shot > 0:
        dataset_root = VISA_FEW_SHOT_DIR
    else:
        dataset_root = VISA_ZERO_SHOT_DIR

    test_img_path = os.path.join(dataset_root, 'Test', category, 'Images')
    train_img_path = os.path.join(dataset_root, 'Train', category, 'Images')
    seed_file = os.path.join(dataset_root, 'Train', category, 'selected_samples_per_run.txt')

    test_img_tot_paths, test_gt_tot_paths, test_tot_labels, \
    test_tot_types = load_phase(test_img_path)

    selected_train_img_tot_paths = []
    selected_train_gt_tot_paths = []
    selected_train_tot_labels = []
    selected_train_tot_types = []

    if k_shot > 0:
        train_img_tot_paths, train_gt_tot_paths, train_tot_labels, \
        train_tot_types = load_phase(train_img_path)
        with open(seed_file, 'r') as f:
            files = f.readlines()
        begin_str = f'{experiment_indx}-{k_shot}: '

        training_indx = []
        for line in files:
            if line.count(begin_str) > 0:
                strip_line = line[len(begin_str):-1]
                index = strip_line.split(' ')
                training_indx = index

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
