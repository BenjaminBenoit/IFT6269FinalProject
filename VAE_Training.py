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
from IWAE import IWAE
from Util import Util
from torch import optim
from Settings import Settings
from torchvision import datasets, transforms


######## CUDA AVAILABILITY CHECK

Util.checkDeviceAndCudaAvailability()


######## LOAD DATA

currentTransform = transforms.ToTensor()

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


######## RUN TRAINING, VALIDATION AND TESTING

Util.runTrainValidTestOnModel(
        modelVAE, 
        optimizer, 
        trainLoader,
        validLoader,
        testLoader,
        Settings.SAVE_INFO_PATH_VAE,
        Settings.SAVE_MODEL_PATH_VAE)
