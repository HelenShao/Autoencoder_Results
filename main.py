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
import data, mean_std, corr_coef, denormalize

####################################### INPUT #########################################
# Data parameters
seed         = 4
mass_per_particle = 6.56561e+11
f_rockstar = "Rockstar_z=0.0.txt"

# Training Parameters
batch_size    = 1
learning_rate = [0.0018567298164386605, 0.0038610487268080844, 0.0026306211435600966, 
                 0.0037704711237611503, 0.007176742303899918, 0.0012538187976255252, 
                 0.0002251745574373147, 0.00023958811694910358, 0.00795278166454001]
weight_decay  = [1.8150136635735584e-05, 7.960078603551047e-05, 2.6906612529208158e-05,
                 8.129414159523295e-05, 3.324908083323674e-05, 2.338655095145669e-05,
                 3.203195072519904e-05, 1.984905444344302e-05, 1.1737004589022407e-05]

# Architecture parameters
input_size    = 11
n_layers      = [5,5,6,5,3,2,3,2,2]
out_features  = [[82, 80, 77, 29], [80, 116, 177, 43], [151, 10, 200, 135, 18], 
                 [189, 77, 82, 82], [164, 91], [48], [185, 196], [104], [158]]
bottleneck    = [2,3,4,5,6,7,8,9,10]

# Model parameters
n_models      = 9
f_best_model  = ['HALOS_AE_135.pt', 'HALOS_AE_117.pt', 'HALOS_AE_206.pt', 'HALOS_AE_201.pt',
                 'HALOS_AE_78.pt', 'HALOS_AE_165.pt', 'HALOS_AE_152.pt', 'HALOS_AE_220.pt',
                 'HALOS_AE_109.pt']

# Use GPUs 
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')
    
    
# Get the denormalized input and output halos
denorm_input, denorm_output = denormalize.denorm(input_size, bottleneck, out_features, 
                                                 n_layers, seed, batch_size, f_rockstar, n_models, device, f_best_model)

# Save denormalized input and output as numpy files for future use
np.save("denorm_input.npy", denorm_input)
np.save("denorm_output.npy", denorm_output)

# Shape of data:
# 9 different predictions 
# Each of the 9 has 367 halos
# Each halo has 11 properties

# denorm_output[1][1] --> the 2nd halo prediction (bottleneck 3) and the first halo (it has 11 properties)
