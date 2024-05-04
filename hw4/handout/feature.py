import csv
import numpy as np
import argparse

VECTOR_LEN = 300  # Length of glove vector
MAX_WORD_LEN = 64  # Max word length in dict.txt and glove_embeddings.txt


################################################################################
# We have provided you the functions for loading the tsv and txt files. Feel   #
# free to use them! No need to change them at all.                             #
################################################################################


def load_tsv_dataset(file):
    """
    Loads raw data and returns a tuple containing the reviews and their ratings.

    Parameters:
        file (str): File path to the dataset tsv file.

    Returns:
        An np.ndarray of shape N. N is the number of data points in the tsv file.
        Each element dataset[i] is a tuple (label, review), where the label is
        an integer (0 or 1) and the review is a string.
    """
    dataset = np.loadtxt(file, delimiter='\t', comments=None, encoding='utf-8',
                         dtype='l,O')
    return dataset


def load_feature_dictionary(file):
    """
    Creates a map of words to vectors using the file that has the glove
    embeddings.

    Parameters:
        file (str): File path to the glove embedding file.

    Returns:
        A dictionary indexed by words, returning the corresponding glove
        embedding np.ndarray.
    """
    glove_map = dict()
    with open(file, encoding='utf-8') as f:
        read_file = csv.reader(f, delimiter='\t')
        for row in read_file:
            word, embedding = row[0], row[1:]
            glove_map[word] = np.array(embedding, dtype=float)
    return glove_map


def trim_words(dataset, glove_map):
    trimmed_dataset = []
    for label, review in dataset:
        words = review.split()
        trimmed_review = []
        for word in words:
            if word in glove_map:
                trimmed_review.append(word)
        trimmed_dataset.append((label, trimmed_review))
    return trimmed_dataset


def vectorize_features(trimmed_dataset, glove_map):
    features = []
    for label, trimmed_review in trimmed_dataset:
        sum_vector = np.zeros(VECTOR_LEN)
        for word in trimmed_review:
            sum_vector += glove_map[word]
        if len(trimmed_review) > 0:
            avg_vector = sum_vector / len(trimmed_review)
        else:
            avg_vector = sum_vector
        features.append((label, avg_vector))
    return features


def write_file(features, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for label, feature_vector in features:
            label_str = f'{label:.6f}'
            feature_str = '\t'.join([f'{feature:.6f}' for feature in feature_vector])
            line = f"{label_str}\t{feature_str}\n"
            f.write(line)


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("validation_input", type=str, help='path to validation input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the input .tsv file')
    parser.add_argument("feature_dictionary_in", type=str,
                        help='path to the GloVe feature dictionary .txt file')
    parser.add_argument("train_out", type=str,
                        help='path to output .tsv file to which the feature extractions on the training data should be written')
    parser.add_argument("validation_out", type=str,
                        help='path to output .tsv file to which the feature extractions on the validation data should be written')
    parser.add_argument("test_out", type=str,
                        help='path to output .tsv file to which the feature extractions on the test data should be written')
    args = parser.parse_args()

    train_input = args.train_input
    validation_input = args.validation_input
    test_input = args.test_input
    feature_dictionary_in = args.feature_dictionary_in
    train_out = args.train_out
    validation_out = args.validation_out
    test_out = args.test_out

    train_data = load_tsv_dataset(train_input)
    validation_data = load_tsv_dataset(validation_input)
    test_data = load_tsv_dataset(test_input)
    feature_dictionary = load_feature_dictionary(feature_dictionary_in)

    trimmed_train_data = trim_words(train_data, feature_dictionary)
    trimmed_validation_data = trim_words(validation_data, feature_dictionary)
    trimmed_test_data = trim_words(test_data, feature_dictionary)

    train_data_features = vectorize_features(trimmed_train_data, feature_dictionary)
    validation_data_features = vectorize_features(trimmed_validation_data, feature_dictionary)
    test_data_features = vectorize_features(trimmed_test_data, feature_dictionary)

    write_file(train_data_features, train_out)
    write_file(test_data_features, test_out)
    write_file(validation_data_features, validation_out)
