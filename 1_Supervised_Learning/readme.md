# Supervised Learning

## Introduction

Supervised learning is a fundamental branch of machine learning where algorithms learn from labeled data to make predictions or decisions. Let's start by understanding the primary tasks in supervised learning.

## Tasks in Supervised Learning

Supervised learning involves two primary tasks:

1. **Classification**: In classification tasks, the goal is to predict a discrete class label or category for input data. Common examples include spam detection (classifying emails as spam or not spam), image classification (identifying objects in images), and sentiment analysis (determining the sentiment of text as positive, negative, or neutral).

2. **Regression**: Regression tasks aim to predict a continuous numeric value based on input features. Examples of regression tasks include predicting house prices based on property attributes, estimating the age of a person from their biometric data, and forecasting stock prices.

Now, let's delve into the concepts of parametric and nonparametric models and how they relate to supervised learning.

## Parametric vs. Nonparametric Models

In supervised learning, models can be categorized as either parametric or nonparametric based on their characteristics:

### Parametric Models

- **Assumptions**: Parametric models make specific assumptions about the functional form of the underlying data distribution. These assumptions are often related to the relationship between input features and the target variable.
- **Fixed Parameters**: They have a fixed number of parameters that are determined based on these assumptions.
- **Examples of Parametric Models**:
  1. **Linear Regression**: Assumes a linear relationship between input features and the target variable. It aims to predict a continuous numeric value based on input features.
  2. **Logistic Regression**: Used for binary classification, assumes a logistic function to model the probability of outcomes.

### Nonparametric Models

- **Assumptions**: Nonparametric models do not make strong assumptions about the functional form of the data distribution. They are more flexible in capturing complex patterns.
- **Adaptability**: They can adapt to complex relationships in the data and have a flexible number of parameters, often growing with the size of the data.
- **Examples of Nonparametric Models**:
  1. **k-Nearest Neighbors (KNN)**: Classifies data based on the majority class among its k-nearest neighbors without assuming a specific data distribution. It can also perform regression tasks by predicting a continuous value.
  2. **Decision Trees**: Build a tree structure to make decisions without specifying a functional form. They can handle both classification and regression.
  3. **Random Forests**: An ensemble of decision trees, capable of handling complex data distributions in both classification and regression tasks.
  4. **Support Vector Machines (SVM) with Non-Linear Kernels**: While SVM is originally parametric with a linear kernel, it becomes nonparametric when non-linear kernels (e.g., polynomial or radial basis function) are used, allowing it to model complex decision boundaries in classification and regression (non-linear) tasks.

Understanding the characteristics and trade-offs between these parametric and nonparametric models is essential when choosing an appropriate approach for supervised learning tasks.
