---
title: Unveiling Single Neuron Logistic Regression for Probabilistic Binary Classification
author: Lucas Moreira
date: December 12, 2023
---

# Unveiling Single Neuron Logistic Regression for Probabilistic Binary Classification

In this comprehensive guide, we'll explore **Single Neuron Logistic Regression**, a powerful technique for tackling binary classification problems with a probabilistic approach. We'll delve into the motivation behind this approach, the intricacies of the sigmoid activation function, the binary cross-entropy loss function, and the gradient calculations for training the model.

## The Need for Probabilistic Binary Classification

Traditional binary classification models, such as the perceptron, assume that data is linearly separable. However, real-world data often doesn't adhere to this assumption. Consider the cancer dataset, where data points may overlap, making linear separation impossible. To handle such complex scenarios, we turn to probabilistic binary classification.

## Designing a Single Neuron for Probability Prediction

Instead of directly assigning class labels, we'll construct a single neuron model capable of predicting class probabilities. This paradigm shift opens new avenues for handling complex datasets.

### The Sigmoid Activation Function

At the core of logistic regression is the **sigmoid activation function**:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

The sigmoid function maps any real-valued number \(z\) to a range between 0 and 1. This makes it ideal for generating probabilities, as it ensures that predictions fall within this range.

## Understanding the Binary Cross-Entropy Loss Function

For binary classification tasks, the **binary cross-entropy loss function** is commonly used. It's defined as follows:

$$
L(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^{N} [y^{(i)}\log \hat{y}^{(i)} + (1-y^{(i)}) \log (1 - \hat{y}^{(i)})]
$$

Here's a breakdown of the components:
- \(\mathbf{w}\) represents the weight vector.
- \(b\) is the bias term.
- \(N\) denotes the number of samples in the dataset.
- \(y^{(i)}\) is the true label of the \(i\)-th sample.
- \(\hat{y}^{(i)}\) is the predicted probability of the \(i\)-th sample.

## Navigating the Gradient for Training

To train our logistic regression model effectively, we need to compute the gradient of the loss function with respect to the weights and bias terms. The partial derivatives are as follows:

$$
\frac{\partial L}{\partial w_1} = (\hat{y}^{(i)} - y^{(i)}) x^{(i)}
$$

$$
\frac{\partial L}{\partial b} = (\hat{y}^{(i)} - y^{(i)})
$$

These gradients guide us in adjusting the model's parameters during training.

## Training Our Logistic Regression Neuron

With the gradients in hand, we embark on the training journey. The training process involves iteratively updating the weights and biases using techniques like stochastic gradient descent (SGD). We repeat this process for multiple epochs until the model converges.

## Concluding Remarks

Single neuron logistic regression emerges as a versatile technique for probabilistic binary classification. It shines in scenarios where data is not linearly separable, allowing us to make predictions with associated probabilities, providing more nuanced insights into the data.

## Further Exploration

- Experiment with different learning rates and observe their impact on convergence and performance.
- Extend your analysis by utilizing multiple features from the cancer dataset, exploring their influence on model accuracy.
- Visualize the decision boundary and monitor the cost function's behavior throughout the training process.

---

This guide has unveiled the power of Single Neuron Logistic Regression for probabilistic binary classification. It equips you with the knowledge and tools to handle complex data and make informed decisions with probabilistic predictions.

## Practical Example

In the following example, we'll apply Single Neuron Logistic Regression to the cancer dataset. We'll train a model to predict whether a tumor is malignant or benign based on its radius. Check out the [notebook](/1_Supervised_Learning/6_Single_Neuron_Logistic_Regression_and_Classification/single_neuron.ipynb) for the full code.
