#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 10:48:15 2018


Grid Search for Hyper Parameters in IWAE model 
(include VAE with NUMBER_OF_GAUSSIAN_SAMPLERS_FOR_IWAE = 1)


@author: nicholas
"""


######## IMPORT
import torch
from IWAE import IWAE
from Util import Util
from torch import optim
from Settings import Settings
from torchvision import datasets, transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler



######## HYPER PARAMETERS TO BE TESTED

BATCH_SIZE = [64, 128, 256]
LEARNING_RATE = [0.1, 0.001, 0.00001]
DIMENSION_OF_Z = [2, 50, 100]
NUMBER_OF_GAUSSIAN_SAMPLERS = [1, 5, 50]           # 1 corresponds to VAE



######## SPLITTIG HERE (not in Util.py) FOR NOW. 
######## TODO: change it in Util.py to add batch_size and the dependencies (trainingIWAE.py & trainingVAE.py)

def splitTrainSet(train_dataset, batch_size, ratio=0.1):
    """
    Further splitting our training dataset into training and validation sets.
    In: A Pytorch Dataset object and a ratio for valid size 
    Out: Dataloader for train and Dataloader for valid
    
    """
    # Define the indices
    indices = list(range(len(train_dataset)))      # start with all the indices in training set
    split = int(len(train_dataset) * ratio)        # define the split size
            
    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))
    
    # Contiguous split
    # train_idx, validation_idx = indices[split:], indices[:split]
    
    # define our samplers -- we use a SubsetRandomSampler because it will return
    # a random subset of the split defined by the given indices without replacement
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)
    
    # Create the train_loader -- use your real batch_size which you
    # I hope have defined somewhere above
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                               batch_size=batch_size, 
                                               sampler=train_sampler)
    
    # You can use your above batch_size or just set it to 1 here.  Your validation
    # operations shouldn't be computationally intensive or require batching.
    validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                    batch_size=batch_size, 
                                                    sampler=validation_sampler)
    
    return train_loader, validation_loader    




######## CUDA AVAILABILITY CHECK

Util.checkDeviceAndCudaAvailability()


######## LOAD DATA

currentTransform = transforms.ToTensor()

currentTrainingDataset = datasets.MNIST(Settings.DATASET_PATH, train=True, download=False, transform=currentTransform)
currentTestingDataset = datasets.MNIST(Settings.DATASET_PATH, train=False, download=False, transform=currentTransform)

kwargs = {}
if(Settings.DEVICE == "cuda"):
    kwargs = {'num_workers': Settings.NUMBER_OF_WORKERS, 'pin_memory': Settings.PIN_MEMORY}




"""

For loops to test all combinations of hyper parameters

"""

for batch_size in BATCH_SIZE:
    for lr in LEARNING_RATE:
        for z_size in DIMENSION_OF_Z:
            for sample_size in NUMBER_OF_GAUSSIAN_SAMPLERS:             

                # Split the train set in 90% train set and 10% valid set)
                trainLoader, validLoader = splitTrainSet(currentTrainingDataset, batch_size, ratio=0.1)
                testLoader = torch.utils.data.DataLoader(currentTestingDataset, batch_size=batch_size, shuffle=True, **kwargs)
                
                
                ######## CREATE MODEL
                modelIWAE = IWAE(sample_size, z_size).to(Settings.DEVICE)
                optimizer = optim.Adam(modelIWAE.parameters(), lr=lr)
                
                
                ######## CREATE SAVING FILES PATHS
                SAVE_INFO_PATH = "Savings/iwae_Info_bs{}_lr{}_z{}_k{}.npy".format(batch_size, lr, z_size, sample_size)
                SAVE_MODEL_PATH = "Savings/iwae_StateDict_bs{}_lr{}_z{}_k{}.pth".format(batch_size, lr, z_size, sample_size)
                
                
                ######## RUN TRAINING, VALIDATION AND TESTING
                Util.runTrainValidTestOnModel(
                        modelIWAE, 
                        optimizer, 
                        trainLoader,
                        validLoader,
                        testLoader,
                        SAVE_INFO_PATH,
                        SAVE_MODEL_PATH)
                
                
