import argparse
import numpy as np

class Node:
    def __init__(self, depth=0, feature=None, labels=None):
        self.left = None
        self.right = None
        self.feature = feature
        self.depth = depth
        self.labels = labels
        self.vote = self.get_majority_vote(labels)

    def get_majority_vote(self, labels):
        if len(labels) == 0:
            return None
        # Convert labels to integers
        int_labels = labels.astype(int)
        label_counts = np.bincount(int_labels)
        if len(label_counts) > 1 and label_counts[0] == label_counts[1]:
            return 1
        else:
            return np.argmax(label_counts)

def getEntropy(labels):
    if len(labels) == 0:
        return 0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

def get_mutual_information(data, labels):
    total_entropy = getEntropy(labels)
    attribute_values, counts = np.unique(data, return_counts=True)
    conditional_entropies = np.array([getEntropy(labels[data == val]) for val in attribute_values])
    total_conditional_entropies = np.sum((counts / counts.sum()) * conditional_entropies)
    return total_entropy - total_conditional_entropies

def build_tree(data, labels, feature_names, current_depth, max_depth):
    unique_labels, counts = np.unique(labels, return_counts=True)
    node = Node(depth=current_depth, labels=labels)

    if current_depth == max_depth:
        return node

    mutual_infos = np.array([get_mutual_information(data[:, i], labels) for i in range(data.shape[1])])
    best_attr = np.argmax(mutual_infos)

    if mutual_infos[best_attr] <= 0:
        return node

    node.feature = best_attr
    attr_values = np.unique(data[:, best_attr])

    if len(attr_values) > 1:
        left_data, left_labels = data[data[:, best_attr] == attr_values[0]], labels[data[:, best_attr] == attr_values[0]]
        right_data, right_labels = data[data[:, best_attr] == attr_values[1]], labels[data[:, best_attr] == attr_values[1]]
        node.left = build_tree(left_data, left_labels, feature_names, current_depth + 1, max_depth)
        node.right = build_tree(right_data, right_labels, feature_names, current_depth + 1, max_depth)
    else:
        node.left = build_tree(data, labels, feature_names, current_depth + 1, max_depth)

    return node

def predict(tree, instance):
    if tree.left is None and tree.right is None:
        return tree.vote
    if instance[tree.feature] == 0:
        return predict(tree.left, instance)
    else:
        return predict(tree.right, instance)

def calculate_error(tree, data, labels):
    predictions = np.array([predict(tree, instance) for instance in data])
    return np.mean(predictions != labels)

def print_tree(node, file, feature_names, indent="", parent_feature=None, feature_value=None):
    if node is None:
        return

    zeros = np.sum(node.labels == 0)
    ones = np.sum(node.labels == 1)
    node_info = f"[{zeros} 0/{ones} 1]"

    if parent_feature is not None:
        file.write(f"{indent}{feature_names[parent_feature]} = {feature_value}: {node_info}\n")
    else:
        file.write(f"{indent}{node_info}\n")

    if node.left is not None or node.right is not None:
        if node.left:
            print_tree(node.left, file, feature_names, indent + "| ", node.feature, 0)
        if node.right:
            print_tree(node.right, file, feature_names, indent + "| ", node.feature, 1)

def write_predictions(tree, data, file_path):
    with open(file_path, "w") as f:
        for instance in data:
            f.write(f"{predict(tree, instance)}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, help='path to output .txt file to which the predictions on the training data should be written')
    parser.add_argument("test_out", type=str, help='path to output .txt file to which the predictions on the test data should be written')
    parser.add_argument("metrics_out", type=str, help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str, help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()

    train_data = np.loadtxt(args.train_input, delimiter="\t", skiprows=1)
    test_data = np.loadtxt(args.test_input, delimiter="\t", skiprows=1)

    with open(args.train_input, 'r') as f:
        feature_names = f.readline().strip().split('\t')[:-1]

    train_labels = train_data[:, -1]
    test_labels = test_data[:, -1]
    train_features = train_data[:, :-1]
    test_features = test_data[:, :-1]

    tree = build_tree(train_features, train_labels, feature_names, 0, args.max_depth)

    write_predictions(tree, train_features, args.train_out)
    write_predictions(tree, test_features, args.test_out)

    with open(args.metrics_out, "w") as metrics_file:
        train_error = calculate_error(tree, train_features, train_labels)
        test_error = calculate_error(tree, test_features, test_labels)
        metrics_file.write(f"error(train): {train_error}\n")
        metrics_file.write(f"error(test): {test_error}\n")

    with open(args.print_out, "w") as print_file:
        print_tree(tree, print_file, feature_names)