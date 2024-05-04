
import numpy as np
import argparse

from numpy import ndarray


def sigmoid(x : np.ndarray):
    """
    Implementation of the sigmoid function.

    Parameters:
        x (np.ndarray): Input np.ndarray.

    Returns:
        An np.ndarray after applying the sigmoid function element-wise to the
        input.
    """
    e = np.exp(x)
    return e / (1 + e)


def train(
    theta : np.ndarray, # shape (D,) where D is feature dim
    X : np.ndarray,     # shape (N, D) where N is num of examples
    y : np.ndarray,     # shape (N,)
    num_epoch : int,
    learning_rate : float
) -> ndarray:
    # TODO: Implement `train` using vectorization
    X = np.hstack((np.ones((X.shape[0], 1)), X))

    for epoch in range(num_epoch):
        for i in range(X.shape[0]):
            xi = X[i]
            yi = y[i]
            z = np.dot(xi, theta)
            prediction = sigmoid(z)
            error = prediction - yi
            gradient = xi * error
            theta -= learning_rate * gradient

    return theta


def predict(
    theta : np.ndarray,
    X : np.ndarray
) -> np.ndarray:
    # TODO: Implement `predict` using vectorization
    X = np.hstack((np.ones((X.shape[0], 1)), X))
    predictions = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        z = np.dot(X[i], theta)
        probability = sigmoid(z)
        if probability >= 0.5:
            predictions[i] = 1
        else:
            predictions[i] = 0

    return predictions


def compute_error(
    y_pred : np.ndarray,
    y : np.ndarray
) -> float:
    # TODO: Implement `compute_error` using vectorization
    return np.mean(y_pred != y)

def load_tsv_dataset(file):
    data = np.loadtxt(file, delimiter='\t')
    y = data[:, 0]
    X = data[:, 1:]
    return X, y

def write_labels(predictions, file_path):
    with open(file_path, 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")

def write_metrics(train_error, test_error, file_path):
    with open(file_path, 'w') as f:
        f.write(f"error(train): {train_error:.6f}\n")
        f.write(f"error(test): {test_error:.6f}\n")


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the learning rate, you can use `args.learning_rate`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to formatted training data')
    parser.add_argument("validation_input", type=str, help='path to formatted validation data')
    parser.add_argument("test_input", type=str, help='path to formatted test data')
    parser.add_argument("train_out", type=str, help='file to write train predictions to')
    parser.add_argument("test_out", type=str, help='file to write test predictions to')
    parser.add_argument("metrics_out", type=str, help='file to write metrics to')
    parser.add_argument("num_epoch", type=int,
                        help='number of epochs of stochastic gradient descent to run')
    parser.add_argument("learning_rate", type=float,
                        help='learning rate for stochastic gradient descent')
    args = parser.parse_args()

    X_train, y_train = load_tsv_dataset(args.train_input)
    X_test, y_test = load_tsv_dataset(args.test_input)

    init_theta = np.zeros(X_train.shape[1] + 1)

    theta = train(init_theta, X_train, y_train, args.num_epoch, args.learning_rate)

    train_predictions = predict(theta, X_train)
    test_predictions = predict(theta, X_test)

    train_error = compute_error(train_predictions, y_train)
    test_error = compute_error(test_predictions, y_test)

    write_labels(train_predictions, args.train_out)
    write_labels(test_predictions, args.test_out)
    write_metrics(train_error, test_error, args.metrics_out)

