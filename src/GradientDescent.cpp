#include <iostream>
#include <vector>

// Structure to hold a data point with features and a target value
template <size_t N>
struct DataPoint {
  double features[N];  // Array of features
  double target;       // Target value
};

// Structure to hold model parameters: slope coefficients and intercept
template <size_t N>
struct Parameter {
  double slope[N] = {0};  // Initialize slope to zero
  double intercept = 0;   // Initialize intercept to zero

  // Method to print the slope and intercept
  void print() const {
    std::cout << "slope: [";
    for (int i = 0; i < N; i++) {
      if (i != N - 1) {
        std::cout << slope[i] << " ";
      } else {
        std::cout << slope[i];
      }
    }
    std::cout << "]" << std::endl;
    std::cout << "intercept: [" << intercept << "]" << std::endl;
  }
};

// Class implementing Gradient Descent algorithm
template <size_t N>
class GradientDescent {
 private:
  double learning_rate;  // Learning rate for the gradient descent
  int iterations;        // Number of iterations for the descent
 public:
  // Constructor to initialize learning rate and number of iterations
  GradientDescent(double learning_rate, int iterations)
      : learning_rate(learning_rate), iterations(iterations) {}

  // Predict the target value for a given data point using current parameters
  double predict(const DataPoint<N> &data_point,
                 const Parameter<N> &parameter) const {
    double predicted = parameter.intercept;
    for (int i = 0; i < N; i++) {
      predicted += data_point.features[i] * parameter.slope[i];
    }
    return predicted;
  }

  // Fit the model to the provided data points using gradient descent
  Parameter<N> fit(const std::vector<DataPoint<N>> &data_points) const {
    Parameter<N> parameter;
    size_t data_size = data_points.size();  // Get the number of data points

    // Perform gradient descent for the specified number of iterations
    for (int i = 0; i < iterations; i++) {
      // Update each slope coefficient
      for (int j = 0; j < N; j++) {
        double slope_gradient = 0;
        // Calculate the gradient for the current slope
        for (int k = 0; k < data_size; k++) {
          slope_gradient +=
              (data_points[k].target - predict(data_points[k], parameter)) *
              data_points[k].features[j];
        }
        slope_gradient = slope_gradient / data_size;
        parameter.slope[j] += learning_rate * slope_gradient;
      }

      // Calculate the gradient for the intercept
      double intercept_gradient = 0;
      for (int k = 0; k < data_size; k++) {
        intercept_gradient +=
            data_points[k].target - predict(data_points[k], parameter);
      }
      intercept_gradient = intercept_gradient / data_size;
      parameter.intercept += learning_rate * intercept_gradient;
    }
    return parameter;
  }
};

int main() {
  // Vector of training data points formatted as follows:
  // Each element is a DataPoint containing:
  //   - features: An array of input values (in this case, a single feature)
  //   - target: The corresponding output value
  std::vector<DataPoint<1>> data_points = {
      {{1.0}, 15.0}, 
      {{2.0}, 25.0},  
      {{3.0}, 35.0}, 
      {{4.0}, 45.0},
      {{5.0}, 55.0}, 
      {{6.0}, 65.0},  
      {{7.0}, 75.0}, 
      {{8.0}, 85.0},
      {{9.0}, 95.0}, 
      {{10.0}, 105.0}};
  GradientDescent<1> gradient_descent(0.01, 1000);
  Parameter parameter = gradient_descent.fit(data_points);
  parameter.print();
  return 0;
}
