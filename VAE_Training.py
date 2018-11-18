# -*- coding: utf-8 -*-
"""

IFT6269 - AUTUMN 2018
PROJECT

"""
######## DESCRIPTION
# This file give the training results for VAE model on MNIST


######## IMPORT
import torch
from IWAE import VAE
from Util import Util
from torch import optim
from Settings import Settings
from torchvision import datasets, transforms



######## MAIN


Util.checkDeviceAndCudaAvailability()

currentTransform = transforms.ToTensor()

# First time running this code, it is needed to put the download value to True instead of False
# Ensure that an empy folder named 'data' is created in the same folder as this file source code
currentTrainingDataset = datasets.MNIST(Settings.DATASET_PATH, train=True, download=False, transform=currentTransform)
currentTestingDataset = datasets.MNIST(Settings.DATASET_PATH, train=False, download=False, transform=currentTransform)

kwargs = {}
if(Settings.DEVICE == "cuda"):
    kwargs = {'num_workers': Settings.NUMBER_OF_WORKERS, 'pin_memory': Settings.PIN_MEMORY}

trainLoader = torch.utils.data.DataLoader(currentTrainingDataset, batch_size=Settings.TRAINING_BATCH_SIZE, shuffle=True, **kwargs)
testLoader = torch.utils.data.DataLoader(currentTestingDataset, batch_size=Settings.TESTING_BATCH_SIZE, shuffle=True, **kwargs)

modelVAE = VAE().to(Settings.DEVICE)
optimizer = optim.Adam(modelVAE.parameters(), lr=Settings.LEARNING_RATE)

for indexEpoch in range(1, Settings.NUMBER_OF_EPOCH+1):
    Util.train(modelVAE, trainLoader, optimizer, indexEpoch)
    Util.test(modelVAE, testLoader, indexEpoch)



    