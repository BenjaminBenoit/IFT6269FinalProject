# -*- coding: utf-8 -*-
"""

IFT6269 - AUTUMN 2018
PROJECT

"""
######## DESCRIPTION
# This file give the training results for VAE model on MNIST


######## IMPORT
import numpy as np
from IWAE import IWAE
from Util import Util
from Settings import Settings


######## MAIN

traningDataset = np.loadtxt(Settings.TRAINING_DATASET_PATH)
testingDataset = np.loadtxt(Settings.TESTING_DATASET_PATH)

modelVAE = IWAE()

print("Todo : training on MNIST")
print("Todo : testing on MNIST")

