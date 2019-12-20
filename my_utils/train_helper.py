#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 10:34:35 2019

@author: picost
"""
import os
import numpy as np
import pandas as pd
import torch


VALIDL = 'valid_loss'
TRAINL = 'train_loss'
ACCU = 'accuracy'
EPOCH = 'epoch'

def train_model(model, loaders, criterion, optimizer, n_epochs=10, device=None, 
          train_dir='.', save_name='model.pt', history_file='train_hist.csv'):
    """Returns a trained model
    
    Args:
        
        model : torch nn.Module model to be trained
        loaders (dict): dict containing data loader to train and validate model
        criterion : torch loss function to be used to train model
        optimizer : torch optimizer to be used to train model
        n_epoch (int, optional): number of training epochs. Default is 10.
        device (str, optional): device where the model is traied. 'cuda' or 'cpu'
        train_dir (str, optional): directory where to save/load model and history.
            Default is local dir.
        save_name (str, optional): name after which the model is saved in the
            training directory.
        history_file (str, optional): name after which the history csv file is
            written in the training directory
            
    Return:
        
        model as trained. 
        
    Side effect:
        
        The model for which the validation loss decreased for the last time is
        saved to the disc as well as the training history.
    
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # initialize tracker for minimum validation loss
    history_path = os.path.join(train_dir, history_file)
    cols = [TRAINL, VALIDL, ACCU]
    try:
        existing_hist = pd.read_csv(history_path, index_col=EPOCH)
        e_start = existing_hist.index[-1] + 1
        epochs = np.arange(e_start, e_start + n_epochs, dtype=np.int64)
        new_hist = pd.DataFrame(np.zeros((n_epochs, len(cols))), columns=cols,
                           index=epochs)
        history = existing_hist.append(new_hist, verify_integrity=True)
    except FileNotFoundError:    
        e_start = 1
        epochs = np.arange(e_start, e_start + n_epochs, dtype=np.int64)
        history = pd.DataFrame(np.zeros((n_epochs, len(cols))), columns=cols,
                               index=epochs)
    valid_loss_min = np.Inf 
    for epoch in epochs:
        print('Starting epoch [{}]'.format(epoch))
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        correct = 0
        ###################
        # train the model #
        ###################
        model.train()
        n_train = 0
        for batch_idx, (data, target) in enumerate(loaders['train']):
            optimizer.zero_grad()
            # move to GPU is used
            data, target = data.to(device), target.to(device)
            output = model.forward(data)
            loss = criterion(output, target)
            # backprop
            loss.backward()
            # weight update
            optimizer.step()
            train_loss += loss.item() * data.shape[0] # default loss reduction is 'mean'
            n_train += data.shape[0]
        train_loss /= n_train
        history.loc[epoch][TRAINL] = train_loss
        print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))
        ######################    
        # validate the model #
        ######################
        model.eval()
        n_valid = 0
        for batch_idx, (data, target) in enumerate(loaders['valid']):
            data, target = data.to(device), target.to(device)
            ## update the average validation loss
            output = model(data)
            loss = criterion(output, target)
            valid_loss += loss.item() * data.shape[0]
            top_score, top_class = output.topk(1, dim=1)
            prediction = top_class.view_as(target)
            correct += (prediction == target).type(torch.FloatTensor).sum()
            n_valid += data.shape[0]
        valid_loss /= n_valid
        history.loc[epoch][VALIDL] = valid_loss
        history.loc[epoch][ACCU] = correct / n_valid
        print('Epoch: {} \tValidation Loss: {:.6f}\tAccuracy: {:.6f}'.format(
            epoch, 
            valid_loss,
            history.loc[epoch][ACCU]
            ))
        if valid_loss < valid_loss_min:
            print('Saving model...')
            if not os.path.exists(train_dir):
                print("Create directory: [{}]".format(train_dir))
                os.mkdir(train_dir)
            save_path = os.path.join(train_dir, save_name)
            torch.save(model.state_dict(), save_path)
            history.to_csv(history_path, index_label=EPOCH)
    history.to_csv(history_path, index_label=EPOCH)
    # return trained model
    return model