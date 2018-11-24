# -*- coding: utf-8 -*-
"""

IFT6269 - AUTUMN 2018
PROJECT
IWAE and VAE implementation

"""

######## DESCRIPTION
# This file implement an IWAE
# Since the VAE is a special case of IWAE for which there is only 1 GaussianSampler,
# Choice has been made to implement only one class for both the VAE and IWAE
# This way it will be easier to maintain and comment the code


######## IMPORT
import torch
from torch import nn
import torch.nn.functional as Functional


######## MODEL

# See 5.Experiment from Kingma publication
# Encoder and Decoder have the same number of hidden layers
# Parameters are updated using Stochastic gradient ascent 
# The gradient is computed by differentiating the lower bound estimator
# Plus a weight decay corresponding to a prior p(theta)
# Inherit the nn.Module class which implement methods such as train and eval
class IWAE(nn.Module):
    
    def __init__(self, numberOfGaussianSampler):
        
        # Initialization of neuralNetwork.Module from pytorch
        super(IWAE, self).__init__()
        
        self.inputLayer = nn.Linear(784,400)
        self.meanLayer = nn.Linear(400,20)
        self.varianceLayer = nn.Linear(400,20)
        self.decoderHiddenLayer = nn.Linear(20,400)
        self.outputLayer = nn.Linear(400,784)
        
        # Number of time we will sample from a gaussian
        self.numberOfGaussianSampler = numberOfGaussianSampler
     

    # Encoder output the mean and the covariance of the latent variables distribution
    # This distribution is an approximation of the intractable posterior P(z|X)
    # The encoder will infer P(z|X) using Q(z|X), Q being a simpler distribution (here it's a Gaussian)        
    def encode(self, x):
        inputLayerOutput = Functional.relu(self.inputLayer(x))
        return self.meanLayer(inputLayerOutput), self.varianceLayer(inputLayerOutput)
    
    
    # It's not possible to sample directly from the encoder output
    # a.k.a it's not possible to get z directly from the mean and the variance outputed by the encoder
    # Because by doing so, it will not be possible to do gradient descent (sampling operation doesn't have gradient)
    # To solve this issue, the reparameterization trick will put the non-differentiable operation out of the network
    # Therefore ii will be possible to do gradient descent through the network
    # Reparameterization trick : z = mean(X) + covariance^{1/2}(X) * epsilon
    # epsilon being the result of a sampling being made on a normal Gaussian
    # Return a sample z from the distribution Q(z|X)
    # Unlike the VAE which will do this only once, the IWAE will sample several time
    def sampleZList(self, mu, logvar):
        sampleZList = []
        standardDeviation = torch.exp(logvar / 2)
        for indexGaussianSampler in range(1, self.numberOfGaussianSampler+1):
            epsilon = torch.randn_like(logvar)
            sampleZList.append(mu + standardDeviation * epsilon)
        return sampleZList
    
    
    # Once again unlike the VAE, we will decode several time : one for each sample z
    # Hence the network will generate multiple approximate posterior samples
    # The more sample we have, the more the lower bound approaches the true log-likelihood
    # Return P(X|z)
    def decode(self, zList):
        outputList = []
        for indexGaussianSampler, z in enumerate(zList):
            decoderHiddenLayerOutput = Functional.relu(self.decoderHiddenLayer(z))
            output = torch.sigmoid(self.outputLayer(decoderHiddenLayerOutput))
            outputList.append(output)
        return outputList
    

    # Override the forward method of nn.Module
    # Explicit a way to do a forward pass through the model    
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,784))
        zList = self.sampleZList(mu, logvar)
        return self.decode(zList), mu, logvar
        