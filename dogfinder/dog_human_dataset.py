#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:53:54 2019

@author: picost
"""

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

dog_dir = "/data/picost/ml_samples/dogImages"
human_dir = "/data/picost/ml_samples/lfw"

class DogVSHumanDataset(Dataset):
    
    def __init__(self, dog_files, hum_files, shape=(224, 224)):
        self._dogs = dog_files
        self._humans = hum_files
        self.files = dog_files + hum_files
        self.labels = [0] *  len(dog_files) + [1] * len(hum_files)
        self.shape = shape
        self.class_names = ['dog', 'human']
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except OSError:
            print(img_path)
        size = self.shape
        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                 (0.229, 0.224, 0.225))])
        # discard the transparent, alpha channel (that's the :3) and add the batch dimension
        image = in_transform(image)[:3,:,:]
        return image, self.labels[idx]

## load filenames for human and dog images
#import os
#import numpy as np
#from glob import glob
#human_files = list(glob(os.path.join(human_dir,"*/*")))
#dog_files = list(glob(os.path.join(dog_dir, "*/*/*")))
#
#ds = DogVSHumanDataset(dog_files, human_files)