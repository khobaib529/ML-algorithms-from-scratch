/*
 * Original Python code by Zulkarnine Mahmud.
 * Original Python code : https://github.com/zulkarnine/ds_algo/blob/master/BinaryTreePrinter.py
 * 
 * Printer Utility to print a binary tree like a tree.
 * 
 * This utility allows printing a binary tree in a visual format that resembles a tree structure.
 * The tree is printed with branches connecting nodes, and the nodes are displayed in a clear
 * and readable way.
 * 
 * Can print a binary tree whose root node has at least the following properties:
 * 
 * - node.left    # left child of the node
 * - node.right   # right child of the node
 * - node.repr()  # method to represent the node's value (returns a string)
 * 
 * The printer provides the following functionality:
 * - print_tree(Node* root)  : Prints the binary tree starting from the root node.
 * - get_tree_lines(Node* root) : Returns the lines of the tree as a vector of strings.
 */


#include <string>
#include <vector>
#include <algorithm>

struct NodePrintData {
  NodePrintData(std::vector<std::string> lines, int root_position,
                int root_len)
      : lines(lines),
        root_position(root_position),
        root_len(root_len),
        max_width(0),
        height(lines.size()) {
    for (std::string& line : lines) {
      max_width = std::max<size_t>(line.size(), max_width);
    }
  }
  std::vector<std::string> lines;
  int root_position;
  int root_len;
  int height;
  int max_width;
};

template <typename Node>
class BinaryTreePrinter {
 private:
  char branch_line;
  char left_node_line;
  char right_node_line;
  int extra_padding;

 public:
  BinaryTreePrinter(char branch_line = '.', char left_node_line = '/',
                    char right_node_line = '\\', int extra_padding = 1)
      : branch_line(branch_line),
        left_node_line(left_node_line),
        right_node_line(right_node_line),
        extra_padding(extra_padding) {}

  void print_tree(Node* root) {
    NodePrintData node_data = __treeify(root);
    for (const std::string& line : node_data.lines) {
      std::cout << line << std::endl;
    }
  }

  std::vector<std::string> get_tree_lines(Node* root) {
    NodePrintData node_data = __treeify(root);
    return node_data.lines;
  }
 private:
  NodePrintData __treeify(Node* node) {
    if (node == nullptr) {
      return NodePrintData({}, 0, 0);
    }
    std::string val = node->repr();
    NodePrintData left_node_data = __treeify(node->left);
    NodePrintData right_node_data = __treeify(node->right);
    std::vector<std::string> lines;
    std::string first_line;
    std::string second_line;
    int len_before_val = 0;
    if (left_node_data.max_width > 0) {
      int left_root_end = left_node_data.root_len + left_node_data.root_position;
      int branch_len = left_node_data.max_width - left_root_end;
         
      first_line.append(std::string(left_root_end + 1, ' '));
      first_line.append(std::string(branch_len + extra_padding, branch_line));
      len_before_val = first_line.size();
      second_line.append(std::string(left_root_end, ' '));
      second_line += left_node_line;
      int second_line_size = second_line.size();
      second_line.append(std::string(len_before_val - second_line_size, ' '));
    }

    first_line += val;
    std::string left_padding = "";
    if (right_node_data.max_width > 0) {
      left_padding.append(std::string(val.size() + 1 + extra_padding, ' '));
    }
    if (right_node_data.max_width > 0) {
      first_line.append(std::string(right_node_data.root_position + extra_padding,
                        branch_line));
      second_line.append(
          std::string(right_node_data.root_position + val.size() + extra_padding, ' '));
      second_line += right_node_line;
    }

    lines.push_back(first_line);
    lines.push_back(second_line);
    
    int max_height = std::max(right_node_data.height, left_node_data.height);
    for (int i = 0; i < max_height; ++i) {
      std::string left_line = "";
      std::string right_line = "";
      if (i < left_node_data.height && i < right_node_data.height) {
        left_line = left_node_data.lines[i];
        right_line = right_node_data.lines[i];
      } else if (i < left_node_data.height) {
        left_line = left_node_data.lines[i];
      } else {
        right_line = right_node_data.lines[i];
      }
      std::string line_to_append(left_line);
      line_to_append.reserve(len_before_val + right_node_data.max_width + left_padding.size());

      line_to_append.append(std::string(len_before_val - left_line.size(), ' '));
      line_to_append.append(left_padding);
      line_to_append.append(right_line);
      line_to_append.append(std::string(right_node_data.max_width - right_line.size(), ' '));

      lines.push_back(line_to_append);
    }
    return NodePrintData(lines, len_before_val, val.size());
  }
};
