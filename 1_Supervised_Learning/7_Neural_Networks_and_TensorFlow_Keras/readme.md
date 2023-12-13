---
title: Comprehensive Deep Neural Network (DNN) Guide
author: Lucas Moreira
date: December 12, 2023
---

# Deep Neural Network (DNN) for Image Prediction

In this extensive guide, we will meticulously explore the inner workings of a Deep Neural Network (DNN) with 2 hidden layers, dissecting every component of the model and its training process.

## Introduction to Deep Neural Networks

Deep Neural Networks (DNNs) represent a category of machine learning models that have gained immense popularity for their ability to handle complex tasks, particularly in computer vision and natural language processing.

### Anatomy of a DNN

1. **Input Layer**: The initial layer of the DNN receives raw data. In our case, this layer accommodates 28x28 grayscale images, with each pixel corresponding to a neuron.

2. **Hidden Layers**: Our DNN consists of two hidden layers. These hidden layers are responsible for learning intricate features and patterns within the input data.

3. **Output Layer**: The output layer is where predictions are generated. In image classification tasks, it typically comprises neurons corresponding to the classes we want to predict.

## Forward Propagation

Forward propagation is the foundational process through which data flows within the neural network to produce predictions. It involves calculating the output of each neuron in each layer.

### Mathematical Representation

The output of a neuron is determined by the weighted sum of its inputs passed through an activation function:

$$
z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]}
$$

Where:
- \(z^{[l]}\) represents the weighted sum at layer \(l\).
- \(W^{[l]}\) signifies the weight matrix.
- \(a^{[l-1]}\) denotes the activation from the previous layer.
- \(b^{[l]}\) is the bias for layer \(l\).

The activation function, denoted as \(\sigma^{[l]}\), introduces non-linearity:

$$
a^{[l]} = \sigma^{[l]}(z^{[l]})
$$

### Activation Functions

Activation functions introduce non-linearity into the neural network, enabling it to learn complex relationships. Common activation functions include:

- **Sigmoid Function**: \(\sigma(z) = \frac{1}{1 + e^{-z}}\)
- **ReLU (Rectified Linear Unit) Function**: \(\sigma(z) = \max(0, z)\)
- **Tanh (Hyperbolic Tangent) Function**: \(\sigma(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}\)

## Loss Function

Selecting an appropriate loss function is crucial, as it quantifies the disparity between predicted and actual values.

### Mean Squared Error (MSE) for Regression

For regression tasks, Mean Squared Error (MSE) is commonly employed:

$$
L(y, \hat{y}) = \frac{1}{2m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2
$$

Where:
- \(L\) is the loss.
- \(y^{(i)}\) is the true output for the \(i\)-th example.
- \(\hat{y}^{(i)}\) is the predicted output for the \(i\)-th example.
- \(m\) is the number of examples.

### Cross-Entropy Loss for Classification

In classification tasks, Cross-Entropy Loss is widely used:

$$
L(y, \hat{y}) = - \sum_{i}^{C} y_i \log(\hat{y}_i)
$$

Where:
- \(C\) is the number of classes.
- \(y_i\) is the true label for class \(i\).
- \(\hat{y}_i\) is the predicted probability of class \(i\).

## Backpropagation and Gradient Descent

Training a DNN involves calculating gradients of the loss function with respect to weights and biases using backpropagation. Gradient Descent is then employed to update these parameters and minimize the loss.

### Gradient Descent Update Rule

For each parameter \(w\) and bias \(b\), the update rule is:

$$
w := w - \alpha \frac{\partial L}{\partial w}
$$

$$
b := b - \alpha \frac{\partial L}{\partial b}
$$

Where:
- \(\alpha\) is the learning rate.
- \(\frac{\partial L}{\partial w}\) and \(\frac{\partial L}{\partial b}\) are the gradients.

## Training the DNN

Training a DNN is an iterative process. It involves passing mini-batches of data through the network, calculating gradients, and updating weights and biases to minimize the loss function.

## Testing and Evaluation

After training, the DNN's performance is assessed on a separate test dataset. Common evaluation metrics include accuracy, precision, recall, and F1-score.

## Conclusion

This comprehensive guide has taken you on an exhaustive journey through the intricacies of a Deep Neural Network with 2 hidden layers for image prediction. Understanding each component, from forward propagation to training and evaluation, provides you with a solid foundation in this advanced machine learning technique.

## Practical Example: Image Classification with DNN

In this practical example, we will build a Deep Neural Network (DNN) with 2 hidden layers to classify images of handwritten letters from the A-Z dataset. Click [here](/1_Supervised_Learning/7_Neural_Networks_and_TensorFlow_Keras/DNN.ipynb) to see the notebook. For an example using TensorFlow and Keras, click [here](/1_Supervised_Learning/7_Neural_Networks_and_TensorFlow_Keras/DNN_TensorFlow_Keras.ipynb).
