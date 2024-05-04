import sys
import numpy as np


def majority_vote(input):
    labels = input[:, -1]
    values, counts = np.unique(labels, return_counts=True)
    if counts[0] <= counts[1]:
        return values[1]
    else:
        return values[0]


def predict(majority_vote, input):
    return np.full(input.shape[0], majority_vote)


def getError(prediction, input):
    return np.mean(prediction != input)


def main(train_input, test_input, train_out, test_out, metrics_out):
    train_data = np.genfromtxt(train_input, delimiter='\t', skip_header=1)
    test_data = np.genfromtxt(test_input, delimiter='\t', skip_header=1)

    majority_vote1 = majority_vote(train_data)

    train_prediction = predict(majority_vote1, train_data)
    test_prediction = predict(majority_vote1, test_data)

    train_error = getError(train_prediction, train_data[:, -1])
    test_error = getError(test_prediction, test_data[:, -1])

    with open(train_out, 'w') as file:
        for data in train_prediction:
            file.write(f"{int(data)}\n")

    with open(test_out, 'w') as file:
        for data in test_prediction:
            file.write(f"{int(data)}\n")

    with open(metrics_out, 'w') as metrics_file:
        metrics_file.write(f"error: {train_error}\n")
        metrics_file.write(f"error: {test_error}\n")


if __name__ == "__main__":
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    main(train_input, test_input, train_out, test_out, metrics_out)
