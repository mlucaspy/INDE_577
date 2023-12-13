# Linear Regression

Linear regression is a statistical method used for modeling the relationship between a dependent variable and one or more independent variables. It's one of the simplest yet powerful techniques in statistical modeling and machine learning for predictive analysis.

## Concept of Linear Regression

Linear regression aims to find a linear relationship between a dependent variable (often denoted as $y$) and one or more independent variables (denoted as $x_1, x_2, ..., x_n$). The case with one independent variable is known as simple linear regression, and with more than one independent variable, it's called multiple linear regression.

### Simple Linear Regression

In simple linear regression, we model the relationship between the two variables using a linear equation:

$$ y = mx + c $$

Here:
- $y$ is the dependent variable.
- $x$ is the independent variable.
- $m$ represents the slope of the line, indicating how much $y$ changes for a unit change in $x$.
- $c$ is the y-intercept, indicating the value of $y$ when $x$ is 0.

The goal is to find the best values of $m$ and $c$ that fit the data points. This fit is typically found by minimizing the differences between the observed values and the values predicted by the model.

### Multiple Linear Regression

In multiple linear regression, the model involves several independent variables. The equation extends to:

$$ y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n $$

Where:
- $y$ is the dependent variable.
- $x_1, x_2, ..., x_n$ are independent variables.
- $b_0$ is the y-intercept.
- $b_1, b_2, ..., b_n$ are the coefficients representing the influence of each independent variable on the dependent variable.

### Assumptions of Linear Regression

Linear regression makes several key assumptions:
1. **Linearity**: The relationship between the independent and dependent variables is linear.
2. **Homoscedasticity**: The residuals (differences between observed and predicted values) have constant variance.
3. **Independence**: Observations are independent of each other.
4. **Normal Distribution of Residuals**: For any fixed value of the independent variable, the corresponding residuals are normally distributed.

## Fitting the Model

The process of "fitting" a linear regression model involves finding the values of the coefficients ($m$ and $c$ in simple regression, $b_0, b_1, ..., b_n$ in multiple regression) that minimize the difference between the predicted and actual values. This is commonly done using a method called Ordinary Least Squares (OLS), where the goal is to minimize the sum of the squares of the residuals.

### Ordinary Least Squares

The OLS method seeks to minimize the sum of the squared differences between the observed and predicted values. For simple linear regression, the cost function (also known as the loss function) to be minimized is:

$$ J(m, c) = \frac{1}{n} \sum_{i=1}^{n} (y_i - (mx_i + c))^2 $$

Where $n$ is the number of observations, $y_i$ is the observed value, and $mx_i + c$ is the predicted value. By minimizing this cost function, we can find the optimal values of $m$ and $c$.

## Ordinary Least Squares (OLS) in Linear Regression

In the context of **Linear Regression**, *Ordinary Least Squares* (OLS) is a crucial technique used to estimate the parameters of a linear regression model. It plays a pivotal role in finding the best-fitting line or hyperplane that relates one or more independent variables (predictors) to a dependent variable.

### Linear Regression Model

At the core of linear regression is the idea of modeling the relationship between variables. In its simplest form, we have a simple linear regression model with one independent variable:

$$ Y = \beta_0 + \beta_1 X + \epsilon $$

- $Y$ represents the dependent variable.
- $X$ stands for the independent variable.
- $\beta_0$ is the intercept.
- $\beta_1$ is the slope (coefficient of $X$).
- $\epsilon$ is the error term.

In more complex scenarios, multiple linear regression extends this concept to include multiple independent variables:

$$ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \ldots + \beta_p X_p + \epsilon $$

- $X_1, X_2, \ldots, X_p$ are the independent variables.
- $\beta_0$ represents the intercept.
- $\beta_1, \beta_2, \ldots, \beta_p$ are the coefficients for each independent variable.
- $\epsilon$ denotes the error term.

### OLS Objective

The primary objective of OLS is to estimate the values of $\beta_0, \beta_1, \ldots, \beta_p$ that minimize the sum of squared differences between the observed values of $Y$ and the predicted values based on the model.

### Loss Function

To achieve this goal, OLS defines a loss function (or cost function) that quantifies the discrepancy between the observed and predicted values. For simple linear regression, the loss function is:

$$ L(\beta_0, \beta_1) = \sum_{i=1}^{n} (Y_i - (\beta_0 + \beta_1 X_i))^2 $$

In multiple linear regression, it generalizes to:

$$ L(\beta_0, \beta_1, \ldots, \beta_p) = \sum_{i=1}^{n} (Y_i - (\beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \ldots + \beta_p X_{ip}))^2 $$

The objective is to minimize this loss function with respect to the coefficients ($\beta_0, \beta_1, \ldots, \beta_p$).

### OLS Estimators

To find the OLS estimators for the coefficients, we take the partial derivatives of the loss function with respect to each coefficient and set them equal to zero. The resulting equations lead to the following estimators:

- For the intercept ($\beta_0$):
  $$ \hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X} $$

- For the coefficients ($\beta_1, \ldots, \beta_p$):
  $$ \hat{\beta}_j = \frac{\sum_{i=1}^{n} (X_{ij} - \bar{X}_j)(Y_i - \bar{Y})}{\sum_{i=1}^{n} (X_{ij} - \bar{X}_j)^2} $$

Where:
- $\hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_p$ are the OLS estimators for the coefficients.
- $\bar{Y}$ is the mean of the observed values of $Y$.
- $\bar{X}_j$ is the mean of the values of the $j$-th independent variable.
- $X_{ij}$ represents the value of the $j$-th independent variable for the $i$-th data point.
- $Y_i$ is the observed value of the dependent variable for the $i$-th data point.

These estimators provide the best-fitting linear model that minimizes the sum of squared differences between observed and predicted values, making OLS a valuable tool in regression analysis.


## Practical Example

For a practical example of linear regression, check out this [notebook](/1_Supervised_Learning/2_Linear_Regression_with_Gradient_Descent/Linear_Regression_GD.ipynb) that demonstrates how to implement linear regression using gradient descent.
