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
import numpy as np
import pandas as pd 

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
IWAE_model = IWAE(50, 20)
IWAE_model.load_state_dict(torch.load('Savings/iwae_model_Z20_K50.pth', map_location=device))
VAE_model = IWAE(1, 20)
VAE_model.load_state_dict(torch.load('Savings/vae_model_Z20.pth', map_location=device))


"""
Comparing original input with output

"""

for i in range(6):
    example_in = TestDataset[i][0]
    # From the model, take the image output from tuple (image, mu, logvar), 
    # take the first output from the list of output, reshape, from Pytorch Variable to Numpy
    example_out = IWAE_model(example_in)[0][0].view(1,28,28).detach().numpy()
    Util.printOneMNIST(example_in)
    Util.printOneMNIST(example_out)
    
  
"""
Print generative examples

"""

# Get a sample
sample = [torch.randn(1,20)]

# Decode it

sample_out = IWAE_model.decode(sample)[0][0].view(1,28,28).detach().numpy()


# Print it
Util.printOneMNIST(sample_out)

#Produce a full grid 
n = 10
sample_out_VAE, sample_out_IWAE = np.ones((n*n, 1, 28, 28)), np.ones((n*n, 1, 28, 28)) 

for i in range(n):
    for j in range(n):
        sample = [torch.randn(1, 20)]
        sample_out_VAE[j + i*n] = VAE_model.decode(sample)[0][0].view(1,28,28).detach().numpy()
        sample_out_IWAE[j + i*n] = IWAE_model.decode(sample)[0][0].view(1,28,28).detach().numpy()

print("Samples of VAE exemples: ")     
Util.printGridMNIST(sample_out_VAE, n, save_name = 'MNIST_generate_20D_VAE')
print("Samples of IWAE exemples: ")     
Util.printGridMNIST(sample_out_IWAE, n, save_name = 'MNIST_generate_20D_IWAE')

"""
Print latent space examples 

WARNING: Only works if dimension of z is 2

"""

IWAE_model_2D = IWAE(50, 2)
IWAE_model_2D.load_state_dict(torch.load('Savings/iwae_model_Z2_K50.pth', map_location=device))
VAE_model_2D = IWAE(1, 2)
VAE_model_2D.load_state_dict(torch.load('Savings/vae_model_Z2.pth', map_location=device))


#Produce a full grid 
n = 10
latent_space_VAE, latent_space_IWAE = np.ones((n*n, 1, 28, 28)), np.ones((n*n, 1, 28, 28)) 
grid = np.linspace(-2, 2, n)


for i in range(n):
    for j in range(n):
        sample = [torch.tensor([[grid[i], grid[j]]])]
        latent_space_VAE[j + i*n] = VAE_model_2D.decode(sample)[0][0].view(1,28,28).detach().numpy()
        latent_space_IWAE[j + i*n] = IWAE_model_2D.decode(sample)[0][0].view(1,28,28).detach().numpy()

print("Latent Space of VAE: ")     
Util.printGridMNIST(latent_space_VAE, n, save_name = 'MNIST_Latent_Space_VAE')
print("Latent Space of IWAE: ")     
Util.printGridMNIST(latent_space_IWAE, n, save_name = 'MNIST_Latent_Space_IWAE')

"""
Print dimensionality reduction examples 

WARNING: Only works if dimension of z is 2

"""

print("Dimensionality Reduction of VAE: ")   
Util.MNIST_2d_reduction(VAE_model_2D, TestDataset, save_name = 'MNIST_2d_reduction_VAE')
print("Dimensionality Reduction of IWAE: ")  
Util.MNIST_2d_reduction(IWAE_model_2D, TestDataset, save_name = 'MNIST_2d_reduction_IWAE')

"""
Calculate Negative Log-Likelihood in 2D

WARNING: Need to set Settings.device = 'cpu' and re-import Settings.py
"""

from Settings import Settings

NLL_VAE_2D = Util.calculate_NLL(VAE_model_2D, TestDataset, 100)
NLL_IWAE_2D = Util.calculate_NLL(IWAE_model_2D, TestDataset, 100)

NLL_VAE_20D = Util.calculate_NLL(VAE_model, TestDataset, 100)
NLL_IWAE_20D = Util.calculate_NLL(IWAE_model, TestDataset, 100)

NLL_Results = pd.DataFrame([[NLL_VAE_2D, NLL_IWAE_2D], [NLL_VAE_20D, NLL_IWAE_20D]],
                           columns = ["NLL VAE", "NLL IWAE (k= 50)"],
                           index = ["Dim. Z = 2", "Dim. Z = 20"])

NLL_Results.to_csv("Figures/NLL_results.csv")






