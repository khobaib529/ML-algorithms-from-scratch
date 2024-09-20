# Gradient Descent Algorithm Implementation

This repository provides a C++ implementation of the gradient descent algorithm, which can be used for various machine learning applications, including training models like linear regression and neural networks.

## Overview

The code consists of several key components:

1. **Data Structures**:
   - `DataPoint<N>`: Represents a data point with an array of features and a corresponding target value, where `N` is the number of features.
   - `Parameter<N>`: Stores the model parameters, including slope coefficients and intercept. It includes a method to print these parameters.

2. **Gradient Descent Class**:
   - `GradientDescent<N>`: Implements the gradient descent algorithm for optimizing model parameters. This class features:
     - `predict`: Computes the predicted target value using the current model parameters.
     - `fit`: Adjusts the model parameters by iterating over the dataset and updating slopes and intercepts based on the calculated gradients.

3. **Main Function**:
   - Demonstrates the usage of the `GradientDescent` class with a sample dataset. It fits the model to the data and outputs the final parameters.

## Algorithm

### Gradient Descent

The gradient descent algorithm minimizes a cost function by iteratively updating model parameters in the direction of the steepest descent (negative gradient). Key steps include:

1. **Initialization**: Begin with zero values for slopes and intercept.
2. **Prediction**: Calculate the predicted target based on current parameters.
3. **Gradient Calculation**:
   - For each slope, compute the gradient by averaging the product of the error (difference between actual and predicted values) and the corresponding feature.
   - Calculate the gradient for the intercept by averaging the errors.
4. **Parameter Update**: Update parameters by adding the product of the learning rate and the calculated gradients.
5. **Iteration**: Repeat the process for a specified number of iterations or until convergence.

### Example

The provided example dataset consists of features with corresponding target values. After executing the algorithm, the fitted parameters will approximate the relationship in the data.

## Usage

1. Clone the repository.
2. Compile the code using a C++ compiler (e.g., g++).
3. Run the executable to see the fitted parameters printed to the console.

```bash
g++ -o GradientDescent GradientDescent.cpp
./GradientDescent
