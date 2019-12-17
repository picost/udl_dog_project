# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision import datasets
import helpers

dog_dir = "/data/picost/ml_samples/dogImages"
human_dir = "/data/picost/ml_samples/lfw"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device : [{}]".format(device))

### TODO: Write data loaders for training, validation, and test sets
## Specify appropriate transforms, and batch_sizes

batch_size = 64

# %%

transform = transforms.Compose([
    transforms.Resize(255),
    transforms.RandomHorizontalFlip(), # randomly flip and rotate
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), 
                         (0.229, 0.224, 0.225)),
    ])

# Training Dataset
train_dir = os.path.join(dog_dir, 'train')
train_set = datasets.ImageFolder(train_dir, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
                                           shuffle=True)
# validation Dataset
valid_dir = os.path.join(dog_dir, 'valid')
valid_set = datasets.ImageFolder(valid_dir, transform=transform)
valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, 
                                           shuffle=True)
# Test Dataset
test_dir = os.path.join(dog_dir, 'test')
test_set = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, 
                                          shuffle=True)

# %%

images, labels = next(iter(train_loader))
helpers.imshow(images[0], normalize=True)

# %%
from dog_cnn_classifier_1 import Net

#-#-# You do NOT have to modify the code below this line. #-#-#

# instantiate the CNN
model_scratch = Net()
# move tensors to GPU if CUDA is available
model_scratch.to(device)

# %%



