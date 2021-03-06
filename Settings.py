# -*- coding: utf-8 -*-
"""

This file contains all the Settings (constants) used accross the application

"""


class Settings:
    
    
    # =============================================================== FILE PATH
    DATASET_PATH='./data Fashion'
    
    # file path for vae saving results (only used if SAVE_RESULTS = True)
    SAVE_INFO_PATH_VAE = "Savings/iwae_Info_bs128_lr0.001_z20_k1.npy"
    # file path for saving vae model (only used if SAVE_RESULTS = True)
    SAVE_MODEL_PATH_VAE = "Savings/iwae_StateDict_bs128_lr0.001_z20_k1.pth"
    
    # file path for saving iwae results (only used if SAVE_RESULTS = True)
    SAVE_INFO_PATH_IWAE = "Savings/Fashion_iwae_model_info_Z20_K5.npy"
    # file path for saving iwae model (only used if SAVE_RESULTS = True)
    SAVE_MODEL_PATH_IWAE = "Savings/Fashion_iwae_model_Z20_K5.pth"
    
    
    # ================================================== MODEL RELATED SETTINGS
    TRAINING_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 5000
    TESTING_BATCH_SIZE = 10000
    
    # TODO : check if it is still needed ?
    LOG_INFORMATION_INTERVAL = 1000
    
    NUMBER_OF_EPOCH = 120
    LOSSVARIATION = 0.0001         # Stop training if variation is smaller than it
    
    LEARNING_RATE = 0.001
    
    #Dimension of latent vector Z
    
    DIMENSION_OF_Z = 20
    # Number of gaussian samplers when initializing an IWAE
    NUMBER_OF_GAUSSIAN_SAMPLERS_FOR_IWAE = 5
    
    # An IWAE with only one gaussian sampler is equivalent to a VAE
    # Here it is a constant used to initialize the VAE and it shouldn't be modified
    NUMBER_OF_GAUSSIAN_SAMPLERS_FOR_VAE = 1
    

    # =========================================================== CUDA SETTINGS
    # Whether or not the model should run on the CPU or GPU
    # If DEVICE="cpu" then the model run on CPU
    # If DEVICE="cuda" then the model run on GPU
    DEVICE = "cuda"
    
    # Note : this parameter is only used if DEVICE="cuda"
    # Beware of the memory usage when setting this parameter
    NUMBER_OF_WORKERS = 1
    
    # Note : this parameter is only used if DEVICE="cuda"
    # See documentation : https://pytorch.org/docs/master/notes/cuda.html
    # ('Use pinned memory buffers' part)
    # If execution of the program freeze, put this parameter to false
    PIN_MEMORY = True
    
    
    # ========================================================== OTHER SETTINGS
    # Whether or not after training-validating-testing a model, we want to save results
    # Possible values are True or False
    SAVE_RESULTS = True
    
    
