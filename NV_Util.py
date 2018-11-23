# -*- coding: utf-8 -*-
"""

IFT6269 - AUTOMN 2018
PROJECT
UTILIY CLASS

"""

######## DESCRIPTION
# This file implement utilities methods


######## IMPORT
import torch
from NV_Settings import Settings
import torch.nn.functional as Functional
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


######## UTIL

class Util:

    # Raise an exception if settings device use GPU but cuda is not available
    def checkDeviceAndCudaAvailability():
        if Settings.DEVICE == "cuda" and not torch.cuda.is_available():
            print("Settings.DEVICE specify a GPU computation but CUDA is not available")
            raise
  
    
    
    
    def train(model, trainLoader, optimizer, epoch):
        model.train()
        trainLoss = 0
        qt_data = 0
        
        for batchIndex, (data, target) in enumerate(trainLoader):
            qt_data += len(data)
            data = data.to(Settings.DEVICE)
            # re-initialize the gradient computation
            optimizer.zero_grad()
            posteriorResults, mu, logvar = model(data)
            loss = Util.calculateLoss(posteriorResults, mu, logvar, data)
            loss.backward()
            trainLoss += loss.item()
            optimizer.step()
        
        trainLoss /= qt_data     
        print('\n==> Epoch: {} \n======> Train loss: {:.4f}'.format(
              epoch, trainLoss))
        
        return trainLoss
        
        
        
    def eval(model, Loader, epoch, setType):         # SetType is 'Valid' or 'Test'
        model.eval()
        lossCumul = 0
        qt_data = 0
        
        # No gradient descent is being made during testing
        with torch.no_grad():
            for index, (data, _) in enumerate(Loader):
                qt_data += len(data)
                data = data.to(Settings.DEVICE)
                posteriorResults, mu, logvar = model(data)
                loss = Util.calculateLoss(posteriorResults, mu, logvar, data) 
                lossCumul += loss.item()
                
        lossCumul /= qt_data
        print('======> {} loss: {:.4f}'.format(setType, lossCumul))
        
        return lossCumul
        
    
    
        
    # The objective function is made of two terms
    # First term is the KL divergence between two distributions : Q(z|X) and P(z|X)
    # Q being the distribution of our latent variables and P the true distribution of our data
    # The second term is the Binary cross entropy
    def calculateLoss(posteriorResults, mu, logvar, x):
        binaryCrossEntropy = Functional.binary_cross_entropy(posteriorResults, x.view(-1, 784), reduction='sum')
        klDivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return binaryCrossEntropy + klDivergence


    def saveFigure(fileName, axes):
        print("Todo")
        

    def createGraphic(titleFigure, fileName, data):
        print("Todo")
        
        
        
    
    def splitTrainSet(train_dataset, ratio=0.1):
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
                                                   batch_size=Settings.TRAINING_BATCH_SIZE, 
                                                   sampler=train_sampler)
        
        # You can use your above batch_size or just set it to 1 here.  Your validation
        # operations shouldn't be computationally intensive or require batching.
        validation_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                                        batch_size=Settings.VALID_BATCH_SIZE, 
                                                        sampler=validation_sampler)
        
        return train_loader, validation_loader
     
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        