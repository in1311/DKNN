# DKNN
This is a pytorch implementation of DKNN: deep kriging neural network for interpretable geospatial interpolation
![DKNN framework](https://github.com/in1311/DKNN/assets/140888660/0b1f8538-f751-43f3-85f8-cc72dd7decd2)

# Requirements
* numpy
* datetime
* os
* pandas
* torch
* tensorboard
* sklearn
* matplotlib
* math

# Running examples
python main.py  

You can adjust the parameters:
* datafile:  sampled dataset in folder "Data/dataset"
* batch_size: batch size
* lr:  learning rate
* hidden_neuron:  [input dimension, model dimension, trend dimension]. Note that the input dimension should be equal to the number of all variables (auxiliary and target) in the dataset
* pe_weight:  weight of positional vector 
* top_k:  top k nearest neighbors
* loss_type:  loss function type
* optim_type:  optimizer type
* if_summary:  if save the training summary or not
* if_save_model:  if save the best model or not

Or you can run the demo.ipynb file，which encompasses code blocks for data loading, preprocessing, model definition, training, and predicting, providing a more comprehensive running example.

The train log and results are saved in folder "results"
