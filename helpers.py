#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 15:15:29 2019

@author: picost
"""
import numpy as np
import matplotlib.pyplot as plt

def imshow(image, ax=None, title=None, normalize=True):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    image = image.numpy().transpose((1, 2, 0))
    if normalize:
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean
        image = np.clip(image, 0, 1)
    ax.imshow(image)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(axis='both', length=0)
    ax.set_xticklabels('')
    ax.set_yticklabels('')
    return ax

def conv2d_carac(h_in, w_in, in_chan, out_chan, k_size, stride=1, padding=0, dilatation=1):
    """
    """
    h_out = int(1 + (h_in + 2 * padding - (k_size - 1)*dilatation - 1) / stride)
    w_out = int(1 + (w_in + 2 * padding - (k_size - 1)*dilatation - 1) / stride)
    n_features = h_out * w_out * out_chan
    n_params = k_size**2 * in_chan * out_chan
    return h_out, w_out, n_features, n_params