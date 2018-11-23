# IFT6269 Final Project - VAE and IWAE

## GPU / CPU switch
To run the model on GPU : in Settings.py, put the parameter DEVICE to 'cuda'  
To run the model on CPU : in Settings.py, put the parameter DEVICE to 'cpu'  


## Run experimentation
To get the results of the VAE experimentations, run the VAE_Training.py file
To get the results of the IWAE experimentations, run the IWAE_Training.py file

## To save and load models (and hyperparameters)
Save on GPU, Load on CPU

Save:
* torch.save(model.state_dict(), PATH)

Load:
* device = torch.device('cpu')
* model = TheModelClass(*args, **kwargs)
* model.load_state_dict(torch.load(PATH, map_location=device))


## TODO  
* delete this todo section of the readme before sending source code  
* implement IWAE    
* implement IWAE_Training.py
* run whole dataset
* double check the comments and verify if everything is ok (add, remove, modify if necessary)  
* either put the VAE in its own file (VAE.py) or implement a IWAE which can be initialize as a VAE  


## Methodology
Steps to share the source code :  
* git add .  
* git commit -m "commit message explaining the changes"  
* git pull --rebase
* git push origin master


## Versions  
This code was executed using Spyder as an IDE.

The versions used are the following :  
Python version : 3.6.1  
Numpy version : 1.12.1  
Spyder version : 3.2.4  
Pytorch version : 0.4.1


## Content  
IWAE.py : implementation of the IWAE  
Settings.py : constants used accross the application  
Util.py : methods used accross the application  
VAE_Training.py : training of the VAE on MNIST  
IWAE_Training.py : training of the IWAE on MNIST  

