# -*- coding: utf-8 -*-
"""

IFT6269 - AUTOMN 2018
PROJECT
UTILIY CLASS

"""

######## DESCRIPTION
# This file implement utilities methods


######## IMPORT
from Settings import Settings
import torch.nn.functional as Functional

######## UTIL


class Util:
    
    
    def train(model, trainLoader, epoch):
        model.train()
        
        for batch_idx, (data, target) in enumerate(trainLoader):
            output = model(data)
            loss = Functional.nll_loss(output, target)
            loss.backward()
            if batch_idx % Settings.LOG_INFORMATION_INTERVAL == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(trainLoader.dataset),
                    100. * batch_idx / len(trainLoader), loss.item()))
        
        
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
            

    def saveFigure(fileName, axes):
        print("Todo")
        

    def createGraphic(titleFigure, fileName, data):
        print("Todo")
        
        