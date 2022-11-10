import json
import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class VideoData(Dataset):
    def __init__(self, mode='train', image_set='2'):
        self.root = f'step_images/train/STEP-ICCV21-0{image_set}' if mode == 'train' else f'step_images/test/STEP' \
                                                                                          f'-ICCV21-0{image_set} '
        self.length = 0
        self.dataset = self.load_data()

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img, gt = self.dataset[item]
        return img, gt

    def load_data(self):
        dataset = []
        for root, dirs, files in os.walk(self.root, topdown=False):
            image_files = [f for f in files if f.endswith('.jpg')]
            for file in image_files:
                img_path = os.path.join(root, file)
                # print(img_path)
                gt_path = img_path.replace('.jpg', '.json')
                if not os.path.exists(img_path) or not os.path.exists(gt_path):
                    raise IOError("{} does not exist".format(img_path))
                img = Image.open(img_path).convert('RGB')
                with open(gt_path) as gt_file:
                    gt = np.asarray(json.load(gt_file))
                dataset.append([img, gt])
                self.length += 1
        return dataset
