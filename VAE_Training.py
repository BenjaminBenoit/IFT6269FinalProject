# -*- coding: utf-8 -*-
"""

IFT6269 - AUTUMN 2018
PROJECT

------------------------>>>  Version updated by Nicholas Vachon

"""
######## DESCRIPTION
# This file give the training results for VAE model on MNIST



######## IMPORT
import torch
import numpy as np
from IWAE import IWAE
from Util import Util
from torch import optim
from Settings import Settings
from torchvision import datasets, transforms



Util.checkDeviceAndCudaAvailability()

currentTransform = transforms.ToTensor()


######## LOAD DATA

currentTrainingDataset = datasets.MNIST(Settings.DATASET_PATH, train=True, download=False, transform=currentTransform)
currentTestingDataset = datasets.MNIST(Settings.DATASET_PATH, train=False, download=False, transform=currentTransform)

kwargs = {}
if(Settings.DEVICE == "cuda"):
    kwargs = {'num_workers': Settings.NUMBER_OF_WORKERS, 'pin_memory': Settings.PIN_MEMORY}

# Split the train set in 90% train set and 10% valid set)
trainLoader, validLoader = Util.splitTrainSet(currentTrainingDataset, ratio=0.1)
testLoader = torch.utils.data.DataLoader(currentTestingDataset, batch_size=Settings.TESTING_BATCH_SIZE, shuffle=True, **kwargs)


######## CREATE MODEL

# VAE is a special case of IWAE for which we take only 1 sample z
modelVAE = IWAE(Settings.NUMBER_OF_GAUSSIAN_SAMPLERS_FOR_VAE).to(Settings.DEVICE)
optimizer = optim.Adam(modelVAE.parameters(), lr=Settings.LEARNING_RATE)



######## TRAIN

# Keep loosses at each Epoch [[train], [valid], [test]]
losses = [[],[],[]]
qt_epoch = 0

for indexEpoch in range(1, Settings.NUMBER_OF_EPOCH+1):
    
    losses[0].append(Util.train(modelVAE, trainLoader, optimizer, indexEpoch))
    losses[1].append(Util.eval(modelVAE, validLoader, indexEpoch, "Valid"))
    losses[2].append(Util.eval(modelVAE, testLoader, indexEpoch, "Test"))
    qt_epoch = indexEpoch                   #To be able to save 
    
    # If at least 2 losses 
    if indexEpoch >= 2:
        train_loss_variation = abs(losses[0][-1] - losses[0][-2])
        # Stop if train loss variation between 2 epoch smaller Settings.LOSSVARIATION
        if train_loss_variation < Settings.LOSSVARIATION: 
            break

    
######## SAVE 
 
info = {'qt_epoch': qt_epoch, 'lr': Settings.LEARNING_RATE, \
        'batch_size': Settings.TRAINING_BATCH_SIZE,\
        'losses': losses}       
np.save('Savings/model_info.npy', losses)

# Save trained model
torch.save(modelVAE.state_dict(), "Savings/model_state_dict.pth")
    
    
    
    
    
    
    
