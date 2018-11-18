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
from Settings import Settings
import torch.nn.functional as Functional

######## UTIL

class Util:
    
    def train(model, trainLoader, optimizer, epoch):
        model.train()
        trainLoss = 0
        for batchIndex, (data, target) in enumerate(trainLoader):
            # re-initialize the gradient computation
            optimizer.zero_grad()
            posteriorResults, mu, logvar = model(data)
            loss = Util.calculateLoss(posteriorResults, mu, logvar, data)
            loss.backward()
            trainLoss += loss.item()
            optimizer.step()
            if batchIndex % Settings.LOG_INFORMATION_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batchIndex * len(data), len(trainLoader.dataset),
                    100. * batchIndex / len(trainLoader), loss.item()))
                
        print('====> Epoch: {} Average loss: {:.4f}'.format(
              epoch, trainLoss / len(trainLoader.dataset)))
        
        
    def test(model, testLoader, epoch):
        model.eval()
        testLoss = 0
        
        # No gradient descent is being made during testing
        with torch.no_grad():
            for index, (data, _) in enumerate(testLoader):
                posteriorResults, mu, logvar = model(data)
                loss = Util.calculateLoss(posteriorResults, mu, logvar, data)             
                testLoss += loss.item()

        testLoss /= len(testLoader.dataset)
        print('====> Test set loss: {:.4f}'.format(testLoss))
        
        
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
        
        