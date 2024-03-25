# DKNN
This is a pytorch implementation of DKNN: deep kriging neural network for interpretable geospatial interpolation

# Requirements
* numpy
* datetime
* os
* pandas
* torch
* sklearn
* matplotlib
* math

# Training
python main.py  

You can adjust the parameters:
* datafile:  sampled dataset in folder "Data/random_field/dataset"
* batch_size: batch size
* lr:  learning rate
* hidden_neuron:  [input dimension, model dimension, trend dimension]
* pe_weight:  weight of positional vector 
* top_k:  top k nearest neighbors
* loss_type:  loss function type
* optim_type:  optimizer type
* if_summary:  if save the training summary or not
* if_save_model:  if save the best model or not

The train log and results are saved in folder "results"
