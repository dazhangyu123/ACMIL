import random

import h5py
import numpy as np
import torch
from timm.utils import accuracy
from sklearn.model_selection import train_test_split
import pandas as pd
from torchvision.datasets import ImageFolder
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import os
from PIL import Image
import sys
import json

def split_dataset_camelyon(file_path, conf):
    h5_data = h5py.File(file_path, 'r')
    split_file_path = './splits/%s/split_%s.json'%(conf.dataset, conf.seed)
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        slide_names = list(h5_data.keys())
        train_val_names, test_names = [], []
        for name in slide_names:
            if 'test' in name:
                test_names.append(name)
            else:
                train_val_names.append(name)
        train_names, val_names = train_test_split(train_val_names, test_size=0.1)
    train_split, val_split, test_split = {}, {}, {}
    for (names, split) in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for name in names:
            slide = h5_data[name]

            label = slide.attrs['label']
            feat = slide['feat'][:]
            coords = slide['coords'][:]

            split[name] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names



def split_dataset_bracs(file_path, conf):
    csv_path = './dataset_csv/bracs.csv'
    slide_info = pd.read_csv(csv_path).set_index('slide_id')
    class_transfer_dict_3class = {0:0, 1:0, 2:0, 3:1, 4:1, 5:2, 6:2}
    class_transfer_dict_2class = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1}

    h5_data = h5py.File(file_path, 'r')
    slide_names = list(h5_data.keys())
    train_split, val_split, test_split = {}, {}, {}
    train_names, val_names, test_names = [], [], []
    for slide_id in slide_names:
        slide = h5_data[slide_id]

        label = slide.attrs['label']
        if conf.n_class == 3:
            label = class_transfer_dict_3class[label]
        elif conf.n_class == 2:
            label = class_transfer_dict_2class[label]

        feat = slide['feat'][:]
        coords = slide['coords'][:]


        split_info = slide_info.loc[slide_id]['split_info']
        if split_info == 'train':
            train_names.append(slide_id)
            train_split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
        elif split_info == 'val':
            val_names.append(slide_id)
            val_split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
        else:
            test_names.append(slide_id)
            test_split[slide_id] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names



def split_dataset_lct(file_path, conf):
    # csv_path = './dataset_csv/lct.csv'
    # slide_info = pd.read_csv(csv_path).set_index('slide_id')
    class_transfer_dict_4class = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 3}
    class_transfer_dict_2class = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    h5_data = h5py.File(file_path, 'r')
    split_file_path = './splits/%s/split_%s.json' % (conf.dataset, conf.seed)
    if os.path.exists(split_file_path):
        with open(split_file_path, 'r') as json_file:
            data = json.load(json_file)
        train_names, val_names, test_names = data['train_names'], data['val_names'], data['test_names']
    else:
        slide_names = list(h5_data.keys())

        # train_val_names, test_names = [], []
        # for name in slide_names:
        #     split_info = slide_info.loc[name]['split_info']
        #     if split_info == 'test':
        #         test_names.append(name)
        #     else:
        #         train_val_names.append(name)
        train_val_names, test_names = train_test_split(slide_names, test_size=0.2)
        train_names, val_names = train_test_split(train_val_names, test_size=0.25)

    train_split, val_split, test_split = {}, {}, {}
    for (names, split) in [(train_names, train_split), (val_names, val_split), (test_names, test_split)]:
        for name in names:
            slide = h5_data[name]
            label = slide.attrs['label']

            if conf.n_class == 4:
                label = class_transfer_dict_4class[label]
            elif conf.n_class == 2:
                label = class_transfer_dict_2class[label]

            if conf.B > 1:
                feat = np.zeros([conf.n_patch, conf.feat_d])
                coords = 0
                n = min(slide['feat'][:].shape[0], conf.n_patch)
                feat[:n] = slide['feat'][:]

            else:

                feat = slide['feat'][:]
                coords = 0

            split[name] = {'input': feat, 'coords': coords, 'label': label}
    h5_data.close()
    return train_split, train_names, val_split, val_names, test_split, test_names



class HDF5_feat_dataset2(object):
    def __init__(self, data_dict, data_names):
        self.data_dict = data_dict
        self.data_names = data_names

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """

        return self.data_dict[self.data_names[index]]

class HDF5_feat_dataset3(object):
    def __init__(self, file_path, data_names):
        self.data_names = data_names
        self.file_path = file_path

    def __len__(self):
        return len(self.data_names)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """

        return torch.load(os.path.join(self.file_path, self.data_names[index]+ '.pth'))




def generate_fewshot_dataset(train_split, train_names, num_shots):
    if num_shots < len(train_names) and num_shots > 0:
        labels = [it['label'] for it in train_split.values()]
        train_split_ = {}
        train_names_ = []
        for l in set(labels):
            indices = [index for index, element in enumerate(labels) if element == l]
            selected_indices = random.sample(indices, num_shots)
            names = [train_names[index] for index in selected_indices]
            train_names_ += names
            split = {name: train_split[name] for name in names}
            train_split_.update(split)
        return train_split_, train_names_
    else:
        return train_split, train_names


def build_HDF5_feat_dataset(file_path, conf):
    if conf.dataset == 'camelyon':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_camelyon(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'bracs':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_bracs(file_path, conf)
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)
    elif conf.dataset == 'lct':
        train_split, train_names, val_split, val_names, test_split, test_names = split_dataset_lct(file_path, conf)
        # save_dir = 'splits/%s' % conf.dataset
        # os.makedirs(save_dir, exist_ok=True)
        # json.dump({'train_names': train_names, 'val_names': val_names, 'test_names': test_names},
        #           open(os.path.join(save_dir, 'split_%s.json' % conf.seed), 'w'))
        # sys.exit()
        train_split, train_names = generate_fewshot_dataset(train_split, train_names, num_shots=conf.n_shot)
        return HDF5_feat_dataset2(train_split, train_names), HDF5_feat_dataset2(val_split, val_names), HDF5_feat_dataset2(test_split, test_names)







if __name__ == '__main__':
    split_dataset_lct('/mnt/Xsky/zyl/dataset/lct/roi_feats_backbone_Resnet50_pretrain_natural_supervised')
