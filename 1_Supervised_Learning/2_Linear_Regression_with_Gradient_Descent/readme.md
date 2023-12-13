# Understanding Linear Regression with Gradient Descent

Linear Regression is a fundamental algorithm in the field of machine learning and statistics, used for predicting a continuous outcome variable (dependent variable) based on one or more predictor (independent) variables. The method aims to find the best-fitting straight line through the points of the dataset. This blog post will delve into linear regression, focusing on its implementation through Gradient Descent.

## Basics of Linear Regression

### The Linear Model

In its simplest form (simple linear regression), the model predicts the outcome variable $Y$ as a linear combination of the predictor variable $X$. This relationship is represented as:

$$ Y = \beta_0 + \beta_1X + \epsilon $$

where:
- $\beta_0$ is the y-intercept.
- $\beta_1$ is the slope of the line.
- $\epsilon$ is the error term.

### Cost Function: Mean Squared Error (MSE)

The cost function for linear regression is usually the Mean Squared Error (MSE), which measures the average of the squares of the errors, i.e., the average squared difference between the observed actual outcoming values and the values predicted by the model. It is given by:

$$ MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $$

where:
- $y_i$ is the actual value.
- $\hat{y}_i$ is the predicted value.
- $n$ is the number of observations.

## Gradient Descent

Gradient Descent is an optimization algorithm used to minimize the cost function in machine learning algorithms. It is used to find the values of the parameters $\beta_0$ and $\beta_1$ that minimize the cost function.

### The Gradient Descent Algorithm

The algorithm starts with initial guesses for the values of $\beta_0$ and $\beta_1$ and iteratively updates these values in the direction that reduces the MSE.

#### Update Rules

The values of $\beta_0$ and $\beta_1$ are updated as follows:

$$ \beta_0 := \beta_0 - \alpha \frac{\partial}{\partial \beta_0} MSE(\beta_0, \beta_1) $$
$$ \beta_1 := \beta_1 - \alpha \frac{\partial}{\partial \beta_1} MSE(\beta_0, \beta_1) $$

where $\alpha$ is the learning rate, a hyperparameter that controls how much we adjust the weights with respect to the loss gradient.

#### Gradient Computation

The partial derivatives of the MSE with respect to $\beta_0$ and $\beta_1$ are given by:

$$ \frac{\partial}{\partial \beta_0} MSE(\beta_0, \beta_1) = \frac{2}{n} \sum_{i=1}^{n} - (y_i - (\beta_0 + \beta_1 x_i)) $$
$$ \frac{\partial}{\partial \beta_1} MSE(\beta_0, \beta_1) = \frac{2}{n} \sum_{i=1}^{n} - x_i(y_i - (\beta_0 + \beta_1 x_i)) $$

### Convergence

The algorithm converges to the minimum value when the cost function reaches a plateau or starts increasing, indicating that the best possible values for the parameters have been found.

## Advanced Concepts in Linear Regression

### Multiple Linear Regression

When there are multiple predictor variables, the model is called Multiple Linear Regression:

$$ Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + ... + \beta_nX_n + \epsilon $$

Gradient Descent can still be applied in a similar manner but with more partial derivatives to compute for each parameter.

### Regularization

To prevent overfitting, regularization techniques like Ridge Regression (L2 regularization) and Lasso Regression (L1 regularization) can be used. These techniques add a penalty to the cost function.

### Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) is a variant of Gradient Descent where the parameters are updated for each training example, which can be faster and can also help avoid local minima.

## Conclusion

Linear Regression with Gradient Descent is a powerful tool in predictive modeling. Understanding the fundamentals of the algorithm and its variations is crucial for effectively applying it to real-world problems.

## Practical Example and Further Reading

For a practical example and further reading on this topic, please visit [click here](/1_Supervised_Learning/2_Linear_Regression_with_Gradient_Descent/Linear_Regression_GD.ipynb).




