#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 17:27:28 2018


Testing traind models on test data:
    
  - Comparing original input with output
  - Generate samples


@author: nicholas
"""


import torch
from IWAE import IWAE
from Settings import Settings
from torchvision import datasets, transforms
from Util import Util

"""
 Load Test Data
 
 
 For MNIST: 
     - TestDataset is organize as a list of tuples (image, value on image):
         image is a tensor(1,28,28)
         value on image is a tensor(1) indicating the value on image
 
"""

currentTransform = transforms.ToTensor()

TestDataset = datasets.MNIST(Settings.DATASET_PATH, train=False, download=False, transform=currentTransform)




"""
Load Model

"""

device = torch.device('cpu')
modelToTest = IWAE(2)
modelToTest.load_state_dict(torch.load('Savings/iwae_model_state_dict.pth', map_location=device))




"""
Comparing original input with output

"""

for i in range(6):
    example_in = TestDataset[i][0]
    # From the model, take the image output from tuple (image, mu, logvar), 
    # take the first output from the list of output, reshape, from Pytorch Variable to Numpy
    example_out = modelToTest(example_in)[0][0].view(1,28,28).detach().numpy()
    Util.printOneMNIST(example_in)
    Util.printOneMNIST(example_out)




"""
Print generative examples

"""


# Get a sample
    
sample = [torch.rand(1,20)]
    
# Decode it

sample_out = modelToTest.decode(sample)[0][0].view(1,28,28).detach().numpy()


# Print it
Util.printOneMNIST(sample_out)



















