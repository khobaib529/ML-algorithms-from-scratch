/*
 * DecisionTreeClassifier.cpp
 *
 * This file implements a Decision Tree classifier from scratch using basic
 * principles of machine learning. The classifier supports binary or multi-class
 * classification based on the well-known algorithms of Information Gain (using
 * Gini Index or Entropy) for feature selection. The tree is built recursively
 * and split based on the feature that maximizes the information gain at each
 * node.
 *
 * Key Components:
 * - `DataPoint`: Represents a single data point with a set of feature values.
 * - `Dataset`: Represents a collection of data points (features and
 * corresponding labels), including methods for splitting the dataset based on
 * feature thresholds.
 * - `BestSplitInfo`: Holds information about the best feature split, including
 * the feature index, threshold, and resulting child datasets.
 * - `Node`: Represents a node in the decision tree, which can either be a
 * decision node (containing a feature index and threshold) or a leaf node
 * (containing a predicted label).
 * - `DecisionTreeClassifier`: The main class that builds and trains a decision
 * tree. It includes methods for fitting the model to the training data,
 * predicting new data, and printing the resulting tree.
 * - `Dataset read_csv`: A helper function to load the Iris dataset (or any
 * similar dataset in CSV format).
 *
 * Key Functionalities:
 * - Building a decision tree using a recursive process based on the best
 * feature split that maximizes information gain.
 * - The ability to calculate information gain using Gini Index or Entropy.
 * - The decision tree can be limited by a maximum depth and a minimum number of
 * samples required to split a node.
 * - A print function to visualize the tree structure.
 * - A simple prediction method that traverses the decision tree to classify new
 * samples.
 *
 * External Dependencies:
 * - "utils/BinaryTreePrinter.h": A utility for printing the tree structure in a
 * readable format.
 * - "utils/csv.h": A utility for reading CSV files (for dataset loading).
 *
 * Example Usage:
 * - The program loads the Iris dataset from a CSV file, builds a decision tree
 * using a maximum depth of 5 and a minimum sample split of 2, prints the tree,
 * and predicts the label of a new test sample.
 *
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <ranges>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/BinaryTreePrinter.h"
#include "utils/csv.h"

// Represents a single data point with its feature values.
struct DataPoint {
  DataPoint(std::vector<double>& features) : features(features) {}
  std::vector<double> features;  // Feature values for a single data point

  double operator[](size_t i) const { return features[i]; }
  size_t size() const { return features.size(); }
};

// Represents a dataset containing feature samples and their corresponding
// labels.
struct Dataset {
  Dataset() = default;
  Dataset(std::vector<DataPoint>& X, std::vector<int32_t>& Y) {
    assert(X.size() ==
           Y.size());  // Ensure X and Y have the same number of samples
    this->X = X;
    this->Y = Y;
  }

  void push_back(DataPoint data_point, int32_t label) {
    X.push_back(data_point);
    Y.push_back(label);
  }

  std::vector<DataPoint>
      X;  // Feature set, with each DataPoint representing a sample
  std::vector<int32_t> Y;  // Labels for each sample

  size_t size() const { return Y.size(); }  // Returns the number of samples

  Dataset split_on_threshold(size_t feature_index, double threshold,
                             bool less_equal) {
    Dataset dataset_split;
    for (size_t i = 0; i < size(); ++i) {
      if (less_equal && X[i][feature_index] <= threshold) {
        dataset_split.push_back(X[i], Y[i]);
        continue;
      }
      if (!less_equal && X[i][feature_index] > threshold) {
        dataset_split.push_back(X[i], Y[i]);
      }
    }
    return dataset_split;
  }
};

// Represents the best feature split for a dataset, including split datasets,
// feature index, threshold, and information gain.
struct BestSplitInfo {
  Dataset dataset_left;  // Subset of data where feature values are <= threshold
  Dataset dataset_right;  // Subset of data where feature values are > threshold
  size_t feature_index;   // Index of the feature used for the split
  double threshold;       // Threshold value for the split on the feature
  double info_gain;       // Information gain achieved by this split
};

template <typename DataType>
std::vector<DataType> unique_values(std::vector<DataType>& values) {
  std::set<DataType> unique_set(values.begin(), values.end());
  return std::vector<DataType>(unique_set.begin(), unique_set.end());
}

double gini_index(std::vector<int32_t>& y) {
  std::vector<int32_t> class_labels = unique_values<int32_t>(y);
  double gini = 1;
  for (int32_t& cls : class_labels) {
    std::size_t count =
        std::ranges::count_if(y, [cls](int32_t value) { return value == cls; });
    double p_cls = static_cast<double>(count) / y.size();
    gini -= std::pow(p_cls, 2);
  }
  return gini;
}

double entropy(std::vector<int32_t>& y) {
  std::vector<int32_t> class_labels = unique_values<int32_t>(y);
  double entropy = 0;
  for (int32_t& cls : class_labels) {
    std::size_t count =
        std::ranges::count_if(y, [cls](int32_t value) { return value == cls; });
    double p_cls = static_cast<double>(count) / y.size();
    entropy += (-p_cls * std::log2(p_cls));
  }
  return entropy;
}

double information_gain(std::vector<int32_t>& parent,
                        std::vector<int32_t>& l_child,
                        std::vector<int32_t>& r_child,
                        std::string mode = "entropy") {
  double weight_l = static_cast<double>(l_child.size()) / parent.size();
  double weight_r = static_cast<double>(r_child.size()) / parent.size();
  double gain = 0;
  if (mode == "gini") {
    gain = gini_index(parent) -
           (weight_l * gini_index(l_child) + weight_r * gini_index(r_child));
  } else {
    gain = entropy(parent) -
           (weight_l * entropy(l_child) + weight_r * entropy(r_child));
  }
  return gain;
}

std::vector<double> get_column(std::vector<DataPoint>& X,
                               size_t feature_index) {
  std::vector<double> result;
  for (DataPoint& features : X) {
    result.push_back(features[feature_index]);
  }
  return result;
}

// Represents a node in the decision tree, containing feature index, threshold,
// child nodes, information gain, and leaf value.
class Node {
 public:
  Node(size_t feature_index = 0, double threshold = 0, Node* left = nullptr,
       Node* right = nullptr, double info_gain = 0, int32_t value = INT32_MIN)
      : feature_index(feature_index),
        threshold(threshold),
        left(left),
        right(right),
        info_gain(info_gain),
        value(value) {}

  std::string repr() {
    if (value == INT32_MIN) {
      std::string str_repr("x_");
      str_repr += std::to_string(feature_index);
      str_repr.append(" >= ");
      str_repr.append(std::to_string(threshold));
      return str_repr;
    }
    return std::to_string(value);
  }

 public:
  size_t feature_index;  // Index of the feature used for the split
  double threshold;      // Threshold value for the split
  Node* left;            // Pointer to the left child node
  Node* right;           // Pointer to the right child node
  double info_gain;      // Information gain from the split
  int32_t value;         // Value for leaf nodes
};

class DecisionTreeClassifier {
 public:
  DecisionTreeClassifier(size_t min_samples_split = 2, uint64_t max_depth = 3)
      : min_samples_split(min_samples_split),
        max_depth(max_depth),
        root(nullptr) {}

  Node* build_tree(Dataset& dataset, uint64_t curr_depth = 0) {
    std::vector<DataPoint>& X = dataset.X;
    std::vector<int32_t>& Y = dataset.Y;
    size_t num_samples = X.size();
    size_t num_features = X.front().size();
    if (num_samples >= min_samples_split && curr_depth <= max_depth) {
      BestSplitInfo best_split =
          get_best_split(dataset, num_samples, num_features);

      if (best_split.info_gain > 0) {
        Node* left_subtree =
            build_tree(best_split.dataset_left, curr_depth + 1);
        Node* right_subtree =
            build_tree(best_split.dataset_right, curr_depth + 1);
        return new Node(best_split.feature_index, best_split.threshold,
                        left_subtree, right_subtree, best_split.info_gain);
      }
    }

    double leaf_value = calculate_leaf_value(Y);
    return new Node(0, 0, nullptr, nullptr, 0, leaf_value);
  }

  BestSplitInfo get_best_split(Dataset& dataset, size_t num_samples,
                               size_t num_features) {
    BestSplitInfo best_split;
    double max_info_gain = std::numeric_limits<double>::lowest();

    for (size_t feature_index = 0; feature_index < num_features;
         ++feature_index) {
      std::vector<double> feature_values = get_column(dataset.X, feature_index);
      std::vector<double> possible_thresholds =
          unique_values<double>(feature_values);

      for (double& threshold : possible_thresholds) {
        Dataset dataset_left =
            dataset.split_on_threshold(feature_index, threshold, true);
        Dataset dataset_right =
            dataset.split_on_threshold(feature_index, threshold, false);

        if (dataset_left.size() > 0 && dataset_right.size() > 0) {
          std::vector<int32_t>& y = dataset.Y;
          std::vector<int32_t>& left_y = dataset_left.Y;
          std::vector<int32_t>& right_y = dataset_right.Y;

          double curr_info_gain = information_gain(y, left_y, right_y, "gini");

          if (curr_info_gain > max_info_gain) {
            best_split.feature_index = feature_index;
            best_split.threshold = threshold;
            best_split.dataset_left = dataset_left;
            best_split.dataset_right = dataset_right;
            best_split.info_gain = curr_info_gain;
            max_info_gain = curr_info_gain;
          }
        }
      }
    }

    return best_split;
  }

  int32_t calculate_leaf_value(std::vector<int32_t>& Y) {
    std::unordered_map<int32_t, size_t> count_map;
    for (int32_t label : Y) {
      count_map[label]++;
    }
    std::pair<int32_t, int> most_frequent = *std::max_element(
        count_map.begin(), count_map.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; });
    return most_frequent.first;
  }

  void print_tree() {
    BinaryTreePrinter<Node> binary_tree_printer;
    binary_tree_printer.print_tree(root);
  }

  void fit(std::vector<DataPoint>& X, std::vector<int32_t>& Y) {
    Dataset dataset(X, Y);
    root = build_tree(dataset);
  }

  std::vector<double> predict(std::vector<DataPoint>& X) {
    std::vector<double> predictions;
    for (DataPoint& x : X) {
      predictions.push_back(make_prediction(x, root));
    }
    return predictions;
  }

  double make_prediction(DataPoint x, Node* tree) {
    if (tree->value != INT32_MIN) {
      return tree->value;
    }
    double feature_val = x.features[tree->feature_index];
    if (feature_val <= tree->threshold)
      return make_prediction(x, tree->left);
    else
      return make_prediction(x, tree->right);
  }

  ~DecisionTreeClassifier() { delete_tree(root); }

  void delete_tree(Node* node) {
    if (node != nullptr) {
      delete_tree(node->left);
      delete_tree(node->right);
      delete node;
    }
  }

 private:
  size_t min_samples_split;
  uint64_t max_depth;
  Node* root;
};

Dataset read_csv(const std::string& file_path) {
  io::CSVReader<5> in(file_path);
  in.read_header(io::ignore_extra_column, "sepal_length", "sepal_width",
                 "petal_length", "petal_width", "species");
  double sepal_length;
  double sepal_width;
  double petal_length;
  double petal_width;
  std::string species;
  std::vector<DataPoint> X;
  std::vector<int32_t> Y;
  while (in.read_row(sepal_length, sepal_width, petal_length, petal_width,
                     species)) {
    std::vector<double> features = {sepal_length, sepal_width, petal_length,
                                    petal_width};
    X.push_back(DataPoint(features));
    if (species == "setosa") {
      Y.push_back(0);
    }
    if (species == "versicolor") {
      Y.push_back(1);
    }
    if (species == "virginica") {
      Y.push_back(2);
    }
  }
  return Dataset(X, Y);
}

int main() {
  Dataset dataset = read_csv("data/iris.csv");
  DecisionTreeClassifier decision_tree_classifier(2, 5);
  decision_tree_classifier.fit(dataset.X, dataset.Y);
  decision_tree_classifier.print_tree();

  std::vector<DataPoint> test_data = {dataset.X[56]};
  std::cout << decision_tree_classifier.predict(test_data)[0] << std::endl;
  return 0;
}
