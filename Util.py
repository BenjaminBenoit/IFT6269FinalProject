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
        
        
    def test(model, testLoader):
        model.eval()
        testLoss = 0
        correctGuesses = 0
        
        for data, target in testLoader:
            output = model(data)
            testLoss += Functional.nll_loss(output, target, reduction='sum').item()
            prediction = output.max(1, keepdim=True)[1]
            correctGuesses += prediction.eq(target.view_as(prediction)).sum().item()
            
        testLoss /= len(testLoader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            testLoss, correctGuesses, len(testLoader.dataset),
            100. * correctGuesses / len(testLoader.dataset)))
        
        
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
        
        