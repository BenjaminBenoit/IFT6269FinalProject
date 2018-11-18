# -*- coding: utf-8 -*-
"""

IFT6269 - AUTUMN 2018
PROJECT
IWAE and VAE implementation

"""

######## DESCRIPTION
# This file implement a VAE and a IWAE


######## IMPORT
import torch
from torch import nn
import torch.nn.functional as Functional


######## MODEL

# Inherit the nn.Module class which implement methods such as train and eval
class IWAE(nn.Module):
    
    
    def __init__(self):
        print("todo")
        
        
        
        

# See 5.Experiment from Kingma publication
# Encoder and Decoder have the same number of hidden layers
# Parameters are updated using Stochastic gradient ascent 
# The gradient is computed by differentiating the lower bound estimator
# Plus a weight decay corresponding to a prior p(theta)
# Minibatch of size 100
# Inherit the nn.Module class which implement methods such as train and eval
class VAE(nn.Module):

    def __init__(self):
        # Initialization of neuralNetwork.Module from pytorch
        super(VAE, self).__init__()
        
        # Neural net with one hidden layer representing Q(z|X)
        self.inputLayer = nn.Linear(784,400)
        self.meanLayer = nn.Linear(400,20)
        self.varianceLayer = nn.Linear(400,20)
        self.decoderHiddenLayer = nn.Linear(20,400)
        self.outputLayer = nn.Linear(400,784)
      
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
    def sampleZ(self, mu, logvar):
        # torch.randn_like return a tensor with the same size as the tensor given in input
        # this tensor is flled with random numbers from a normal distribution with mean 0 and variance 1
        epsilon = torch.randn_like(logvar)
        standardDeviation = torch.exp(logvar / 2)
        return mu + standardDeviation * epsilon
    
    # Return P(X|z)
    def decode(self, z):
        decoderHiddenLayerOutput = Functional.relu(self.decoderHiddenLayer(z))
        return Functional.sigmoid(self.outputLayer(decoderHiddenLayerOutput))
    
    # Override the forward method of nn.Module
    # Explicit a way to do a forward pass through the VAE
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,784))
        z = self.sampleZ(mu, logvar)
        return self.decode(z), mu, logvar
        