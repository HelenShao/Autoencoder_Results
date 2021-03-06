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

def AutoEncoder(input_size, bottleneck, out_features, n_layers, n_models):
    # define the lists containing the encoder and decoder layers
    encoder_layers = []
    decoder_layers = []
        
    for i in range(n_layers):
        if i == 0: # First and last layers of the model
            if i == n_layers - 1:  # if only 1 hidden layer
                # Add encoder input layer 
                encoder_layers.append(nn.Linear(input_size, bottleneck_neurons))
                encoder_layers.append(nn.LeakyReLU(0.2))

                # Add final decoder output layer
                decoder_layers.append(nn.Linear(bottleneck_neurons, input_size))
                # No activation layer here (decoder output)

            else: 
                # Add encoder input layer 
                encoder_layers.append(nn.Linear(input_size, out_features[0]))
                encoder_layers.append(nn.LeakyReLU(0.2))

                # Define in_features to be out_features for decoder
                in_features = out_features[0]

                # Add final decoder output layer
                decoder_layers.append(nn.Linear(in_features, input_size))
                # No activation layer here (decoder output)

        elif i == n_layers - 1: 
            # add the layers adjacent to the bottleneck
            encoder_layers.append(nn.Linear(in_features, bottleneck))
            encoder_layers.append(nn.LeakyReLU(0.2))

            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Linear(bottleneck, in_features))

        else:
            # Add encoder layers
            encoder_layers.append(nn.Linear(in_features, out_features[i]))
            encoder_layers.append(nn.LeakyReLU(0.2))

            # Define in_features to be out_features for decoder
            in_features = out_features[i] 

            # Add decoder layers
            decoder_layers.append(nn.LeakyReLU(0.2))
            decoder_layers.append(nn.Linear(in_features, out_features[i-1]))

    # Reverse order of layers in decoder list
    decoder_layers.reverse()

    # Complete layers list (symmetric)
    layers = encoder_layers + decoder_layers

    # return the model
    return nn.Sequential(*layers)

# Load model
def load_model(input_size, bottleneck, out_features, n_layers, n_models, device, f_best_model):
    model = AutoEncoder(input_size, bottleneck, out_features, n_layers, n_models).to(device)
    if os.path.exists(f_best_model):
        model.load_state_dict(torch.load(f_best_model, map_location=torch.device('cpu')))
    return model 
