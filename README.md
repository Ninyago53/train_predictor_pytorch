# train_predictor_pytorch
This repository contains a PyTorch implementation of a Multi-Layer Perceptron (MLP) for a binary classification task. The model is trained on an embedding dataset and predict a label for each embedding. The training and validation data is split in a 5% validation and 95% training dataset. The embeddings are loaded from two different sources (x_coco and x_laion) and are concatenated and permutated before being split into the training and validation sets.

The model uses a simple architecture with 4 linear layers, dropout layers in between and a sigmoid activation function in the last layer. The optimizer used is Adam with a learning rate of 1e-3.

The repository also includes a Colab notebook to run the model and the train_predictor.py script which contains the training loop and the MLP class.

To run the model, please make sure to have the necessary dependencies installed and to adjust the file paths to the embedding datasets in the train_predictor.py script.
