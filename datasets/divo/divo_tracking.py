#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import sys
sys.path.append('.')
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import numbers
import random
import os
import cv2

from natsort import natsorted

class DIVOTracking_CrossView(Dataset):
    def __init__(self,data_root,id1,id2):

        self.image1_dir = os.path.join(data_root,"images","train",id1)
        self.image2_dir = os.path.join(data_root,"images","train",id2)

        self.image1_names = natsorted(os.listdir(self.image1_dir))
        self.image2_names = natsorted(os.listdir(self.image2_dir))

        self.image_names = self.image1_names + self.image2_names

        self.len_image1 = len(self.image1_names)
        self.len_image2 = len(self.image2_names)

        self.length = self.len_image1 + self.len_image2
        self.track_gt = 
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image1_names)

    def __getitem__(self, idx):

        if idx < len(self.image1_names):
            image_path = os.path.join(self.image1_dir, self.image1_names[idx])
        else:
            image_path = os.path.join(self.image2_dir, self.image2_names[idx-len(self.image1_names)])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if len(image.shape) == 2:
            image = cv2.merge([image, image, image])

        # image = self.transform(image)

        return {'image': image, 'image_name': self.image_names[idx]}

    def image_size(self):
        '''
        H, W
        '''
        image_path = os.path.join(self.image_dir, self.image_names[0])  
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image.shape[-2:]


class DIVOTracking_SingleView(Dataset):
    def __init__(self,data_root,id):

        self.image_dir = os.path.join(data_root,"images","train",id)

        self.image_names = natsorted(os.listdir(self.image_dir))

        self.length = len(self.image_names)
        self.track_gt = 
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        image_path = os.path.join(self.image_dir, self.image_names[idx])

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if len(image.shape) == 2:
            image = cv2.merge([image, image, image])

        # image = self.transform(image)

        return {'image': image, 'image_name': self.image_names[idx]}

    def image_size(self):
        '''
        H, W
        '''
        image_path = os.path.join(self.image_dir, self.image_names[0])  
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        return image.shape[-2:]