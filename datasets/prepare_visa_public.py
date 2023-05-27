# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import os
import shutil

import numpy as np
from PIL import Image


def _mkdirs_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


parser = argparse.ArgumentParser(description='Data preparation')
parser.add_argument('--split-type', default='1cls', type=str, help='1cls, 2cls_highshot, 2cls_fewshot')
parser.add_argument('--data-folder', default='../datasets/VisA_20220922/', type=str,
                    help='the path to downloaded VisA dataset')
parser.add_argument('--save-folder', default='../datasets/VisA_pytorch/', type=str,
                    help='the target path to save the reorganized VisA dataset facilitating data loading in pytorch')
parser.add_argument('--split-file', default='../datasets/VisA_20220922/split_csv/1cls.csv', type=str,
                    help='the csv file to split downloaded VisA dataset')

config = parser.parse_args()

split_type = config.split_type
split_file = config.split_file
data_folder = config.data_folder
save_folder = os.path.join(config.save_folder, split_type)

data_list = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum', 'macaroni1', 'macaroni2', 'pcb1', 'pcb2',
             'pcb3', 'pcb4', 'pipe_fryum']

if split_type == '1cls':
    for data in data_list:
        train_folder = os.path.join(save_folder, data, 'train')
        test_folder = os.path.join(save_folder, data, 'test')
        mask_folder = os.path.join(save_folder, data, 'ground_truth')

        train_img_good_folder = os.path.join(train_folder, 'good')
        test_img_good_folder = os.path.join(test_folder, 'good')
        test_img_bad_folder = os.path.join(test_folder, 'bad')
        test_mask_bad_folder = os.path.join(mask_folder, 'bad')

        _mkdirs_if_not_exists(train_img_good_folder)
        _mkdirs_if_not_exists(test_img_good_folder)
        _mkdirs_if_not_exists(test_img_bad_folder)
        _mkdirs_if_not_exists(test_mask_bad_folder)

    with open(split_file, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            object, set, label, image_path, mask_path = row
            if label == 'normal':
                label = 'good'
            else:
                label = 'bad'
            image_name = image_path.split('/')[-1]
            mask_name = mask_path.split('/')[-1]
            img_src_path = os.path.join(data_folder, image_path)
            msk_src_path = os.path.join(data_folder, mask_path)
            img_dst_path = os.path.join(save_folder, object, set, label, image_name)
            msk_dst_path = os.path.join(save_folder, object, 'ground_truth', label, mask_name)
            shutil.copyfile(img_src_path, img_dst_path)
            if set == 'test' and label == 'bad':
                mask = Image.open(msk_src_path)

                # binarize mask
                mask_array = np.array(mask)
                mask_array[mask_array != 0] = 255
                mask = Image.fromarray(mask_array)

                mask.save(msk_dst_path)
else:
    for data in data_list:
        train_folder = os.path.join(save_folder, data, 'train')
        test_folder = os.path.join(save_folder, data, 'test')
        mask_folder = os.path.join(save_folder, data, 'ground_truth')
        train_mask_folder = os.path.join(mask_folder, 'train')
        test_mask_folder = os.path.join(mask_folder, 'test')

        train_img_good_folder = os.path.join(train_folder, 'good')
        train_img_bad_folder = os.path.join(train_folder, 'bad')
        test_img_good_folder = os.path.join(test_folder, 'good')
        test_img_bad_folder = os.path.join(test_folder, 'bad')

        train_mask_bad_folder = os.path.join(train_mask_folder, 'bad')
        test_mask_bad_folder = os.path.join(test_mask_folder, 'bad')

        _mkdirs_if_not_exists(train_img_good_folder)
        _mkdirs_if_not_exists(train_img_bad_folder)
        _mkdirs_if_not_exists(test_img_good_folder)
        _mkdirs_if_not_exists(test_img_bad_folder)
        _mkdirs_if_not_exists(train_mask_bad_folder)
        _mkdirs_if_not_exists(test_mask_bad_folder)

    with open(split_file, 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            object, set, label, image_path, mask_path = row
            if label == 'normal':
                label = 'good'
            else:
                label = 'bad'
            image_name = image_path.split('/')[-1]
            mask_name = mask_path.split('/')[-1]
            img_src_path = os.path.join(data_folder, image_path)
            msk_src_path = os.path.join(data_folder, mask_path)
            img_dst_path = os.path.join(save_folder, object, set, label, image_name)
            msk_dst_path = os.path.join(save_folder, object, 'ground_truth', set, label, mask_name)
            shutil.copyfile(img_src_path, img_dst_path)
            if label == 'bad':
                mask = Image.open(msk_src_path)

                # binarize mask
                mask_array = np.array(mask)
                mask_array[mask_array != 0] = 255
                mask = Image.fromarray(mask_array)

                mask.save(msk_dst_path)
