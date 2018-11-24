# IFT6269 Final Project - VAE and IWAE

## GPU / CPU switch
To run the model on GPU : in Settings.py, put the parameter DEVICE to 'cuda'  
To run the model on CPU : in Settings.py, put the parameter DEVICE to 'cpu'  


## Run experimentation
To get the results of the VAE experimentations, run the VAE_Training.py file  
To get the results of the IWAE experimentations, run the IWAE_Training.py file  

## To save and load models 
Save on GPU, Load on CPU

Save model:
* torch.save(model.state_dict(), PATH)

Load model:
* device = torch.device('cpu')
* model = TheModelClass(*args, **kwargs)
* model.load_state_dict(torch.load(PATH, map_location=device))


## Load model info (hyperparameters)
* np.load(PATH).item()    
---Bien mettre le .item() pour utiliser comme dict---



## TODO  
* delete this todo section of the readme before sending source code  
* create hyper parameter search for VAE (including convolutional layers option when creating IWAE model)
* plot graph of training, valid losses (accumulated losses are in modele_info.npy)
* utility function to display outputted images
* utility function to generate random outputs
* add progress bar for training to better estimate remaining time
* cleanup code (remove unused import, add-delete comments, remove unused settings, factorize when necessary)


## Methodology
Steps to share the source code :  
* git add .  
* git commit -m "commit message explaining the changes"  
* git pull --rebase
* git push origin master  

Note : if you changed a file but don't want to commit those changes  
(for example, you modified by mistake on file in the Savings folder)  
Before adding the files and before commit, execute the following command :  
git checkout filename  
Warning : this will revert all the changes made in the filename (and filename only)  


## Versions  
This code was executed using Spyder as an IDE.

The versions used are the following :  
Python version : 3.6.1  
Numpy version : 1.12.1  
Spyder version : 3.2.4  
Pytorch version : 0.4.1


## Content  
IWAE.py : implementation of the IWAE & VAE models  
Settings.py : constants used accross the application  
Util.py : methods used accross the application  
VAE_Training.py : training of the VAE on MNIST  
IWAE_Training.py : training of the IWAE on MNIST  
Savings : Where models get saved (model_state_dict.pth) & dict of model parameters (model_info.npy)
