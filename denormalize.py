import numpy as np
import matplotlib.pyplot as plt
import sys, os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
import data
import corr_coef
import model

def denorm(input_size, bottleneck, out_features, n_layers, seed, batch_size, f_rockstar, n_models, device, f_best_model):
 
    #Create datasets
    train_Dataset, valid_Dataset, test_Dataset = data.create_datasets(seed, batch_size, f_rockstar)

    #Create Test Loader
    test_loader  = DataLoader(dataset=test_Dataset,
                              batch_size=batch_size, shuffle=False)
    
    # load halo data and get mean+std
    halo_data = data.get_halo_data()
    mean, std = data.find_mean_std(halo_data)

    # Make containers to denormalize output and input (test set)
    denorm_output = np.zeros((9, 367, 11), dtype = np.float32)
    denorm_input = np.zeros((9, 367, 11), dtype = np.float32)

    # Get the input and output values
    for j in range(9):
        # Load model
        Autoencoder = model.load_model(input_size, bottleneck[j], out_features[j], n_layers[j], n_models, device, f_best_model[j])

        n_halos = int(3674 * 0.1)                                 # Number of halos in test set
        input_array = np.zeros((n_halos, 11), dtype = np.float32) # Container for input
        predicted = np.zeros((n_halos, 11), dtype = np.float32)   # Container for output

        # Predict values
        i = -1 
        for input in test_loader:
            i += 1
            input_array[i] = input
            output = Autoencoder(input)
            predicted[i] = output.detach().numpy()

       ############################# De-Normalize ################################
        for i in range(11):
            denorm_output[j][:,i] = (predicted[:,i] * std[i]) + mean[i]
            denorm_input[j][:,i]  = (input_array[:,i] * std[i]) + mean[i]

        print(np.mean(denorm_output[j][:,0]), np.mean(denorm_input[j][:,0]))

        # Take 10 to the power of m_vir and J_mag
        denorm_output[j][:,0] = 10**(denorm_output[j][:,0])-1
        denorm_output[j][:,6] = 10**(denorm_output[j][:,6])-1

        denorm_input[j][:,0] = 10**(denorm_input[j][:,0])-1
        denorm_input[j][:,6] = 10**(denorm_input[j][:,6])-1
        
    return denorm_input, denorm_output

