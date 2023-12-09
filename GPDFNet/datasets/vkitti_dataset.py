import os
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import cv2

from utils.file_io import read_img, read_disp

from . import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class StereoDataset(Dataset):
    def __init__(self,
                 transform=None,
                 is_vkitti2=True
                 ):
        super(StereoDataset, self).__init__()

        self.transform = transform
        self.save_filename = False

        self.is_vkitti2 = is_vkitti2


        self.samples = []

    def __getitem__(self, index):
        sample = {}

        # file path
        sample_path = self.samples[index]

        if self.save_filename:
            sample['left_name'] = sample_path['left_name']

        sample['left'] = read_img(sample_path['left'])  # [H, W, 3]
        sample['right'] = read_img(sample_path['right'])

        if 'disp' in sample_path and sample_path['disp'] is not None:
            sample['disp'] = read_disp(sample_path['disp'],
                                       vkitti2=self.is_vkitti2
                                       )  # [H, W]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample


class VKITTI2(StereoDataset):
    def __init__(self,
                 data_dir='../../../share/public/dataset/VKITTI2',
                 transform=None,
                 ):
        super(VKITTI2, self).__init__(transform=transform,
                                      is_vkitti2=True,
                                      )

        # total: 21260
        left_files = sorted(glob(data_dir + '/Scene*/*/frames/rgb/Camera_0/rgb*.jpg'))

        for left_name in left_files:
            sample = dict()

            sample['left'] = left_name
            sample['right'] = left_name.replace('/Camera_0/', '/Camera_1/')
            sample['disp'] = left_name.replace('/rgb/', '/depth/').replace('rgb_', 'depth_')[:-3] + 'png'

            self.samples.append(sample)

def build_dataset(args):
    train_transform_list = [transforms.RandomScale(crop_width=args.img_width),
                            transforms.RandomCrop(args.img_height, args.img_width),
                            transforms.RandomColor(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]

    train_transform = transforms.Compose(train_transform_list)

    train_dataset = VKITTI2(transform=train_transform)

    return train_dataset