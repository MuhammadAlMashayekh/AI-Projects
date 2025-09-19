Simple Convolutional Neural Network (CNN) for Shape Recognition
Overview

This project implements a basic Convolutional Neural Network (CNN) from scratch in Python using only NumPy and PIL.
It is designed for shape recognition, classifying images into 10 possible shape classes (e.g., circle, triangle, square, etc.).

The model is kept simple and educational to demonstrate how CNNs work without using TensorFlow or PyTorch.

Features:
1- Custom convolution and pooling layers implemented from scratch.
2- Recognizes shapes instead of handwritten digits.
3- Fully connected neural network for classification.
4- Softmax layer for probability distribution.
5- Can train on a dataset of labeled shape images.
6- Supports saving/loading model weights (parameters.npz).

Algorithm
The CNN works as follows:
1- Convolution – Apply custom filters to extract edges and shape features.
2- ReLU activation – Keep only positive activations.
3- Max Pooling – Downsample feature maps to retain key features.
4- Flattening – Convert feature maps into a 1D vector.
5- Fully Connected Layers – Map features to class scores.
6- Softmax Output – Predict the most likely shape class.
