# Building a Decision Tree Classifier from Scratch

In this comprehensive guide, we will walk through the process of building a decision tree classifier from scratch. Decision trees are powerful and interpretable machine learning models used for classification tasks. We'll go through each step, including data preprocessing, tree splitting, and making predictions.

## Step 1: Data Preprocessing

Before constructing the decision tree, ensure your dataset is prepared for training. Follow these preprocessing steps:

- **Load Data**: Load your dataset, which should include both features (X) and target labels (y).

- **Handle Categorical Data**: If your dataset contains categorical features, consider encoding them into numerical values, such as one-hot encoding.

- **Split Data**: Divide your dataset into training and testing sets to evaluate the model's performance.

## Step 2: Decision Tree Node Structure

To build the decision tree, create a class to represent nodes. Each node should store the following attributes:

- `gini`: The Gini impurity score, which measures node impurity.
- `num_samples`: The number of samples in the node.
- `num_samples_per_class`: A list of counts for each class in the node.
- `predicted_class`: The predicted class for the node.
- `feature_index`: The index of the feature used for splitting.
- `threshold`: The threshold value for the feature split.
- `left`: The left child node.
- `right`: The right child node.

## Step 3: Calculate Gini Impurity

Implement a function to calculate the Gini impurity of a set of samples. The Gini impurity is defined as:

$$
\text{Gini Impurity} = 1 - \sum_{c} P(c)^2
$$

Where:
- \(c\) represents each class.
- \(P(c)\) is the probability of selecting a sample from class \(c\).

## Step 4: Find the Best Split

Create a function to find the best feature and threshold to split the data and minimize impurity. Consider all possible features and thresholds to determine the split that reduces the Gini impurity the most. This function should return the feature index and threshold for the best split.

## Step 5: Build the Decision Tree

Now, it's time to build the decision tree. Implement a recursive function that takes the following parameters:

- `X`: The feature matrix.
- `y`: The target labels.
- `depth`: The current depth of the tree.
- `max_depth`: The maximum depth allowed for the tree.

The function should do the following:

- Calculate the Gini impurity for the current node.
- Check if further splitting is possible (depth < max_depth).
- If yes, find the best split using the "Find the Best Split" function.
- Create left and right child nodes and recursively call itself for both child nodes.
- Return the current node.

## Step 6: Make Predictions

To make predictions, create a function that traverses the decision tree from the root node. Starting at the root, follow the tree's branches based on feature values until you reach a leaf node. The predicted class of the leaf node is the final prediction.

## Step 7: Evaluation and Fine-Tuning

After building the decision tree model, evaluate its performance on the test dataset. You can use metrics such as accuracy, precision, recall, and F1-score. If needed, fine-tune the model by adjusting hyperparameters like maximum depth or minimum samples per leaf.

## Visualizing the Decision Tree

Let's visualize a simple decision tree to understand its structure:

![Decision Tree Example](https://upload.wikimedia.org/wikipedia/commons/f/f3/CART_tree_titanic_survivors.png)

In the example above, each node represents a decision based on a feature, and branches lead to child nodes. The leaf nodes contain the predicted class labels.

## Conclusion

Congratulations! You've successfully built a decision tree classifier from scratch. Decision trees are interpretable models that can be useful for various classification tasks. Further exploration and experimentation with different datasets and parameters will deepen your understanding of decision trees and their applications.

## Practical Example

In the following [notebook](/1_Supervised_Learning/5_Decision_Trees/decision_tree.ipynb), we'll build a decision tree classifier from scratch and use it to classify Wine types.
