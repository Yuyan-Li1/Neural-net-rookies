import json
import os
import random

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import transforms


class TempVideoData(Dataset):
    def __init__(self, mode='train', image_set='2'):
        self.root = f'breakup_set/train' if mode == 'train' else f'breakup_set/test'
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
                gt_path = img_path.replace('.jpg', '.json')
                if not os.path.exists(img_path) or not os.path.exists(gt_path):
                    raise IOError("{} does not exist".format(img_path))
                img = Image.open(img_path).convert('RGB')
                trans = transforms.Compose([transforms.ToTensor()])
                img_tensor = trans(img)
                with open(gt_path) as gt_file:
                    gt = np.asarray(json.load(gt_file))
                dataset.append([img, gt])
                self.length += 1
        return dataset


class VideoData(Dataset):
    def __init__(self, mode='train', transform=None, target_transform=None):
        self.mode = mode
        self.img_dir = f'breakup_set/train' if mode == 'train' else 'breakup_set/test'
        self.transform = transform
        self.target_transform = target_transform
        for root, dirs, files in os.walk(self.img_dir, topdown=False):
            self.image_files = [f for f in files if f.endswith('.jpg')]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = random.randrange(1, 300) if self.mode == 'train' else random.randrange(301, 600)
        img_name = f'02-000{name:03d}.jpg'
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        gt_path = img_path.replace('.jpg', '.json')
        with open(gt_path) as gt_file:
            gt_list = json.load(gt_file)
            gt = [[value for key, value in x.items()] for x in gt_list]
        if self.transform:
            image = self.transform(image)
        return image, gt
