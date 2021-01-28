# Autoencoder_Results
Visualizing the accuracy of features reproduced by halo-property autoencoder

Following up on the training of the autoencoder to reproduce halo properties from reduced dimensions, this repo contains 
files used to analyze the results of the best model.

Important Features:
1. Denormalization of Input/Output - Used to make the scatterplots
2. Scatterplots with r_squared  - used to compare the true vs reproductions and compute the accuracy wrt y=x line
3. Latent Space Mappings  - used to visualize how the input features are organized in the bottleneck space
4. PCA vs Autoencoder Scatterplots - to determine how well each method performs relative to each other
5. Transporting data  - how to print the input, bottleneck, and output (reproduction) of the best trained model and contain these data in single numpy file for future analysis.
