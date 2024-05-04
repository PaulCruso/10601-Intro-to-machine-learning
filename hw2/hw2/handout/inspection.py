import sys
import numpy as np


def majority_vote(input):
    labels = input[:, -1]
    values, counts = np.unique(labels, return_counts=True)
    if counts[0] <= counts[1]:
        return values[1]
    else:
        return values[0]


def predict(majority_vote, label):
    return np.full(label.shape[0], majority_vote)


def getError(prediction, label):
    return np.mean(prediction != label)


def getEntropy(label):
    values, counts = np.unique(label, return_counts=True)
    countsAll = np.sum(counts)
    entropy = ((-counts[0] * np.log2(counts[0]/countsAll)/countsAll) +
               (-counts[1] * np.log2(counts[1]/countsAll)/countsAll))
    return entropy


def main(train_input,train_out):
    train_data = np.genfromtxt(train_input, delimiter='\t', skip_header=1)

    majority_vote1 = majority_vote(train_data)

    train_prediction = predict(majority_vote1, train_data)

    train_error = getError(train_prediction, train_data[:, -1])

    train_entropy = getEntropy(train_data[:, -1])

    with open(train_out, 'w') as file:
        file.write(f"entropy: {train_entropy}\n")
        file.write(f"error: {train_error}\n")


if __name__ == "__main__":
    train_input = sys.argv[1]
    train_out = sys.argv[2]
    main(train_input, train_out)













