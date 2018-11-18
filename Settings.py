# -*- coding: utf-8 -*-
"""

This file contains all the Settings (constants) used accross the application

"""


class Settings:
    
    DATASET_PATH='./data'
    
    TRAINING_BATCH_SIZE = 1
    TESTING_BATCH_SIZE = 1
    
    LOG_INFORMATION_INTERVAL = 10
    
    NUMBER_OF_EPOCH = 1
    
    LEARNING_RATE = 0.001
    
    # Whether or not the model should run on the CPU or GPU
    # If DEVICE="cpu" then the model run on CPU
    # If DEVICE="cuda" then the model run on GPU
    DEVICE = "cpu"
    
    # Note : this parameter is only used if DEVICE="cuda"
    # Beware of the memory usage when setting this parameter
    NUMBER_OF_WORKERS = 1
    
    # Note : this parameter is only used if DEVICE="cuda"
    # See documentation : https://pytorch.org/docs/master/notes/cuda.html
    # ('Use pinned memory buffers' part)
    # If execution of the program freeze, put this parameter to false
    PIN_MEMORY = True