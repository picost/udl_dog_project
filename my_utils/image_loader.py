#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 12:57:45 2019

@author: picost
"""
import numpy as np
from PIL import Image
# Set PIL to be tolerant of image files that are truncated.
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
import requests

def load_image(img_path, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.
       
    Args:
    
        img_path (str): path from where the image has to be loaded
        shape (int, tuple): images are reshaped accordingly.
            
    Return
    
        Tensor: image as torch tensor with 3 channels
       
    '''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    if shape is not None:
        size = shape
        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                 (0.229, 0.224, 0.225))])
    else:
        in_transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.485, 0.456, 0.406), 
                                                 (0.229, 0.224, 0.225))])
    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image