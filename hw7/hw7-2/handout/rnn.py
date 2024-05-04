import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Function
import argparse
from typing import List, Tuple
import time
import matplotlib.pyplot as plt
import numpy as np
from metrics import evaluate
from tqdm import tqdm

# Initialize the device type based on compute resources being used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DO NOT CHANGE THIS LINE OF CODE!!!!
torch.manual_seed(4)


class TextDataset(Dataset):
    def __init__(self, train_input: str, word_to_idx: dict, tag_to_idx: dict, idx_to_tag: dict):
        """
        Initialize the dictionaries, sequences, and labels for the dataset

        :param train_input: file name containing sentences and their labels
        :param word_to_idx: dictionary which maps words (str) to indices (int). Should be initialized to {} 
            outside this class so that it can be reused for test data. Will be filled in by this class when training.
        :param tag_to_idx: dictionary which maps tags (str) to indices (int). Should be initialized to {} 
            outside this class so that it can be reused for test data. Will be filled in by this class when training.
        :param idx_to_tag: Inverse dictionary of tag_to_idx, which maps indices (int) to tags (str). Should be initialized to {} 
            outside this class so that it can be reused when evaluating the F1 score of the predictions later on. 
            Will be filled in by this class when training.
        """
        self.sequences = []
        self.labels = []
        i = 0  # index counter for word dict
        j = 0  # index counter for tag dict

        # for all sequences, convert the words/labels to indices using 2 dicts,
        # append these indices to the 2 lists, and add the resulting lists of
        # word/label indices to the accumulated dataset

        with (open(train_input, 'r') as f):
            sequence_temp = []
            label_temp = []
            for line in f:
                if line != '\n':
                    word = line.strip().split('\t')[0]
                    tag = line.strip().split('\t')[1]
                    if word not in word_to_idx:
                        word_to_idx[word] = i
                        i += 1
                    if tag not in tag_to_idx:
                        tag_to_idx[tag] = j
                        idx_to_tag[j] = tag
                        j += 1
                    sequence_temp.append(word_to_idx[word])
                    label_temp.append(tag_to_idx[tag])
                else:
                    self.sequences.append(torch.tensor(sequence_temp))
                    self.labels.append(torch.tensor(label_temp))
                    sequence_temp = []
                    label_temp = []

    def __len__(self):
        """
        :return: Length of the text dataset (# of sentences)
        """
        return len(self.sequences)

    def __getitem__(self, idx):
        """
        Return the sequence of words and corresponding labels given input index

        :param idx: integer of the index to access
        :return word_tensor: sequence of words as a tensor
        :return label_tensor: sequence of labels as a tensor
        """
        return self.sequences[idx], self.labels[idx]


class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input: nn.Parameter, weight: nn.Parameter, bias: nn.Parameter):
        """
        Manual implementation of a Layer Linear forward computation that 
        also caches parameters for the backward computation. 

        :param ctx: context object to store parameters
        :param input: training example tensor of shape (batch_size, in_features)
        :param weight: weight tensor of shape (out_features, in_features)
        :param bias: bias tensor of shape (out_features)
        :return: forward computation output of shape (batch_size, out_features)
        """
        ctx.save_for_backward(input, weight)
        output = (torch.matmul(input, torch.transpose(weight, 0, 1))
                  + bias.reshape((1, weight.shape[0])))
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Manual implementation of a Layer Linear backward computation 
        using the cached parameters from forward computation

        :param ctx: context object to access stored parameters
        :param grad_output: partial derviative w.r.t Linear outputs (What is the shape?)
        :returns:
            g_input: partial derivative w.r.t Linear inputs (Same shape as inputs)
            g_weight: partial derivative w.r.t Linear weights (Same shape as weights)
            g_bias: partial derivative w.r.t Linear bias (Same shape as bias, remember that bias is 1-D!!!)
        """
        input, weight = ctx.saved_tensors
        g_input = torch.matmul(grad_output, weight)
        g_weight = torch.matmul(torch.transpose(grad_output, 0, 1), input)
        g_bias = torch.sum(grad_output, dim=0)

        return g_input, g_weight, g_bias


class TanhFunction(Function):
    @staticmethod
    def forward(ctx, input):
        """
        Take the Tanh of input parameter

        :param ctx: context object to store parameters
        :param input: Activiation input (output of previous layers)
        :return: output of tanh activation of shape identical to input
        """
        outputTanh = torch.tanh(input)
        ctx.save_for_backward(outputTanh)
        return outputTanh

    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs backward computation of Tanh activation

        :param ctx: context object to access stored parameters
        :param grad_output: partial deriviative of loss w.r.t Tanh outputs
        :return: partial deriviative of loss w.r.t Tanh inputs
        """
        dlh = ctx.saved_tensors[0]
        dl = grad_output * (1 - dlh ** 2)
        return dl


class ReLUFunction(Function):
    @staticmethod
    def forward(ctx, input):
        """
        Takes the ReLU of input parameter

        :param ctx: context object to store parameters
        :param input: Activation input (output of previous layers) 
        :return: Output of ReLU activation with shape identical to input
        """
        ctx.save_for_backward(input)
        return torch.relu(input)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs backward computation of ReLU activation

        :param ctx: context object to access stored parameters
        :param grad_output: partial deriviative of loss w.r.t ReLU output
        :return: partial deriviative of loss w.r.t ReLU input
        """
        input = ctx.saved_tensors[0]
        dl = grad_output.clone()

        input_1d = input.view(-1)
        dl_1d = dl.view(-1)

        for i in range(input.numel()):
            if input_1d[i] < 0:
                dl_1d[i] = 0
        return dl


class Linear(nn.Module):
    def __init__(self, in_features, out_features):
        """
        Initialize the dimensions and the weight and bias matrix for the linear layer.

        :param in_features: units in the input of the layer
        :param out_features: units in the output of the layer
        """

        super(Linear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # See https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        bound = torch.sqrt(1 / torch.tensor([in_features])).item()

        self.weight = nn.Parameter(nn.init.uniform_(
            torch.empty(out_features, in_features), a=-1 * bound, b=bound))
        self.bias = nn.Parameter(nn.init.uniform_(
            torch.empty(out_features), a=-1 * bound, b=bound))

    def forward(self, x):
        """
        Wrapper forward method to call the self-made Linear layer

        :param x: Input into the Linear layer, of shape (batch_size, in_features)
        """
        return LinearFunction.apply(x, self.weight, self.bias)


class Tanh(nn.Module):
    def __init__(self):
        super(Tanh, self).__init__()

    def forward(self, x):
        """
        Wrapper forward method to call the Tanh activation layer

        :param x: Input into the Tanh activation layer
        :return: Output of the Tanh activation layer
        """
        return TanhFunction.apply(x)


class ReLU(nn.Module):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        """
        Wrapper forward method to call the ReLU activation layer

        :param x: Input into the ReLU activation layer
        :return: Output of the ReLU activation layer
        """
        return ReLUFunction.apply(x)


class RNN(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int, activation: str):
        """
        Initialize the embedding dimensions, hidden layer dimensions, 
        hidden Linear layers, and activation.

        :param embedding_dim: integer of the embedding size
        :param hidden_dim: integer of the dimension of hidden layer 
        :param activation: string of the activation type to use (Tanh, ReLU)
        """
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim

        self.linear1 = Linear(embedding_dim, hidden_dim)
        self.linear2 = Linear(hidden_dim, hidden_dim)

        if activation == 'tanh':
            self.activation = Tanh()
        elif activation == 'relu':
            self.activation = ReLU()

    def forward(self, embeds):
        """
        Computes the forward pass for the RNN using the hidden layers
        and the input represented by embeddings. Sets initial hidden state to zeros.

        :param embeds: a batch of training examples converted to embeddings of size (batch_size, seq_length, embedding_dim)
        :returns: 
            outputs: list containing the final hidden states at each sequence length step. Each element has size (batch_size, hidden_dim),
            and has length equal to the sequence length
        """
        (batch_size, seq_length, _) = embeds.shape
        outputs = []
        sequence = torch.zeros(batch_size, self.hidden_dim)

        for i in range(seq_length):
            sequence = self.activation(self.linear1(embeds[:, i, :]) + self.linear2(sequence))
            outputs.append(sequence)

        return torch.stack(outputs, dim=1)


class TaggingModel(nn.Module):
    def __init__(self, vocab_size: int, tagset_size: int, embedding_dim: int,
                 hidden_dim: int, activation: str):
        """
        Initialize the underlying sequence model, activation name, 
        sequence embeddings and linear layer for use in the forward computation.
        
        :param vocab_size: integer of the number of unique "words" in our vocabulary
        :param tagset_size: integer of the number of possible tags/labels (desired output size)
        :param embedding_dim: integer of the size of our sequence embeddings
        :param hidden_dim: integer of the hidden dimension to use in the Linear layer
        :param activation: string of the activation name to use in the sequence model
        """

        super(TaggingModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.rnn = RNN(embedding_dim, hidden_dim, activation)

        self.linear = Linear(hidden_dim, tagset_size)

    def forward(self, sentences):
        """
        Perform the forward computation of the model (prediction), given batched input sentences.

        :param sentences: batched string sentences of shape (batch_size, seq_length) to be converted to embeddings 
        :return tag_distribution: concatenated results from the hidden to out layers (batch_size, seq_len, tagset_size)
        """
        embeddingTowards = self.embedding(sentences)
        rnn1 = self.rnn(embeddingTowards)
        tagLinear = self.linear(rnn1.view(-1, rnn1.shape[2]))
        tag_distribution = tagLinear.view(sentences.size(0), sentences.size(1), -1)

        return tag_distribution


def calc_metrics(true_list, pred_list, tags_dict):
    """
    Calculates precision, recall and f1_score for lists of tags
    You aren't required to implement this function, but it may be helpful
    in modularizing your code.

    :param true_list: list of true/gold standard tags, in index form
    :param pred_list: list of predicted tags, in index form
    :param tags_dict: dictionary of indices to tags
    :return:
        (optional) precision: float of the overall precision of the two lists
        (optional) recall: float of the overall recall of the two lists
        f1_score: float of the overall f1 score of the two lists
    """
    true_list_tags = [tags_dict[i] for i in true_list]
    pred_list_tags = [tags_dict[i] for i in pred_list]
    precision, recall, f1_score = evaluate(true_list_tags, pred_list_tags)
    return precision, recall, f1_score


def train_one_epoch(model, dataloader, loss_fn, optimizer):
    """
    Trains the model for exactly one epoch using the supplied optimizer and loss function

    :param model: model to train 
    :param dataloader: contains (sentences, tags) pairs
    :param loss_fn: loss function to call based on predicted tags (tag_dist) and true label (tags)
    :param optimizer: optimizer to call after loss calculated
    """
    model.train()
    for sentences, tags in dataloader:
        sentences, tags = sentences.to(device), tags.to(device)
        optimizer.zero_grad()
        tag_distribution = model(sentences)

        flattened_predictions = tag_distribution.view(-1, tag_distribution.shape[-1])
        flattened_true_tags = tags.view(-1)

        loss = loss_fn(flattened_predictions, flattened_true_tags)
        loss.backward()
        optimizer.step()


def predict_and_evaluate(model, dataloader, tags_dict, loss_fn):
    """
    Predicts the tags for the input dataset and calculates the loss, accuracy, and f1 score

    :param model: model to use for prediction
    :param dataloader: contains (sentences, tags) pairs
    :param tags_dict: dictionary of indices to tags
    :param loss_fn: loss function to call based on predicted tags (tag_dist) and true label (tags)
    :return:
        loss: float of the average loss over dataset throughout the epoch
        accuracy: float of the average accuracy over dataset throughout the epoch
        f1_score: float of the overall f1 score of the dataset
        all_preds: list of all predicted tag indices
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for sentences, true_tags in dataloader:
            sentences, true_tags = sentences.to(device), true_tags.to(device)
            tag_distributions = model(sentences)
            loss = loss_fn(tag_distributions.view(-1, tag_distributions.size(-1)), true_tags.view(-1))
            total_loss += loss.item()

            _, predicted_tags = torch.max(tag_distributions, dim=-1)

            predicted_tags = predicted_tags.view(-1)
            true_tags = true_tags.view(-1)

            total_correct += (predicted_tags == true_tags).sum().item()
            total_samples += true_tags.size(0)

            all_preds.extend(predicted_tags.cpu().tolist())
            all_true.extend(true_tags.cpu().tolist())

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_samples

    precision, recall, f1_score = calc_metrics(all_true, all_preds, tags_dict)

    return avg_loss, accuracy, f1_score, all_preds


def train(train_dataloader, test_dataloader, model, optimizer, loss_fn,
          tags_dict, num_epochs: int):
    """
    Trains the model for the supplied number of epochs. Performs evaluation on 
    test dataset after each epoch and accumulates all train/test accuracy/losses.

    :param train_dataloader: contains training data
    :param test_dataloader: contains testing data
    :param model: model module to train
    :param optimizer: optimizer to use in training loop
    :param loss_fn: loss function to use in training loop
    :param tags_dict: dictionary of indices to tags
    :param num_epochs: number of epochs to train
    :return:
        train_losses: list of integers (train loss across epochs)
        train_accuracies: list of integers (train accuracy across epochs)
        train_f1s: list of integers (train f1 score across epochs)        
        test_losses: list of integers (test loss across epochs)
        test_accuracies: list of integers (test accuracy across epochs)
        test_f1s: list of integers (test f1 score across epochs)
        final_train_preds: list of tags (final train predictions on last epoch)
        final_test_preds: list of tags (final test predictions on last epoch)
    """
    train_losses, train_accuracies, train_f1s = [], [], []
    test_losses, test_accuracies, test_f1s = [], [], []

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        train_one_epoch(model, train_dataloader, loss_fn, optimizer)

        model.eval()
        train_loss, train_accuracy, train_f1, train_preds = predict_and_evaluate(model, train_dataloader,
                                                                                 tags_dict, loss_fn)
        test_loss, test_accuracy, test_f1, test_preds = predict_and_evaluate(model, test_dataloader, tags_dict, loss_fn)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        train_f1s.append(train_f1)

        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        test_f1s.append(test_f1)

        if epoch == num_epochs - 1:
            final_train_preds, final_test_preds = train_preds, test_preds

    return (train_losses, train_accuracies, train_f1s,
            test_losses, test_accuracies, test_f1s,
            final_train_preds, final_test_preds)


def main(train_input: str, test_input: str, embedding_dim: int,
         hidden_dim: int, num_epochs: int, activation: str):
    """
    Main function that creates dataset/dataloader, initializes the model, optimizer, and loss.
    Also calls training and inferences loops.
    
    :param train_input: string of the training .txt file to read
    :param test_input: string of the testing .txt file to read
    :param embedding_dim: dimension of the input embedding vectors
    :param hidden_dim: dimension of the hidden layer of the model
    :param num_epochs: number of epochs for the training loop
    :param activation: string of the type of activation to use in seq model

    :return: 
        train_losses: train loss from the training loop
        train_accuracies: train accuracy from the training loop
        train_f1s: train f1 score from the training loop
        test_losses: test loss from the training loop
        test_accuracies: test accuracy from the training loop
        test_f1s: test f1 score from the training loop
        train_predictions: final predicted labels from the train dataset
        test_predictions: final predicted labels from the test dataset
    """
    word_to_idx = {}
    tag_to_idx = {}
    idx_to_tag = {}

    # Initialize datasets and dataloaders
    train_dataset = TextDataset(train_input, word_to_idx, tag_to_idx, idx_to_tag)
    test_dataset = TextDataset(test_input, word_to_idx, tag_to_idx, idx_to_tag)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    vocab_size = len(word_to_idx)
    tagset_size = len(tag_to_idx)

    model = TaggingModel(vocab_size, tagset_size, embedding_dim, hidden_dim, activation)

    optimizer = torch.optim.Adam(model.parameters())

    loss_fn = nn.CrossEntropyLoss()

    train_losses, train_accuracies, train_f1s, test_losses, test_accuracies, test_f1s, final_train_preds, final_test_preds = train(
        train_dataloader, test_dataloader, model, optimizer, loss_fn, idx_to_tag, num_epochs
    )

    return train_losses, train_accuracies, train_f1s, test_losses, test_accuracies, test_f1s, final_train_preds, final_test_preds


if __name__ == '__main__':
    # DO NOT MODIFY THIS ARGPARSE CODE
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_dim', type=int, help='size of the embedding vector')
    parser.add_argument('--hidden_dim', type=int, help='size of the hidden layer')
    parser.add_argument('--num_epochs', type=int, help='number of epochs')
    parser.add_argument('--activation', type=str, choices=["tanh", "relu"], help='activation layer to use')
    parser.add_argument('train_input', type=str, help='path to training input .txt file')
    parser.add_argument('test_input', type=str, help='path to testing input .txt file')
    parser.add_argument('train_out', type=str, help='path to .txt file to write training predictions to')
    parser.add_argument('test_out', type=str, help='path to .txt file to write testing predictions to')
    parser.add_argument('metrics_out', type=str, help='path to .txt file to write metrics to')

    args = parser.parse_args()
    # Call the main function
    train_losses, train_accuracies, train_f1s, test_losses, test_accuracies, test_f1s, train_predictions, test_predictions = main(
        args.train_input, args.test_input, args.embedding_dim,
        args.hidden_dim, args.num_epochs, args.activation
    )

    with open(args.train_out, 'w') as f:
        for pred in train_predictions:
            f.write(str(int(pred)) + '\n')
    with open(args.test_out, 'w') as f:
        for pred in test_predictions:
            f.write(str(int(pred)) + '\n')

    train_acc_out = train_accuracies[-1]
    train_f1_out = train_f1s[-1]
    test_acc_out = test_accuracies[-1]
    test_f1_out = test_f1s[-1]

    with open(args.metrics_out, 'w') as f:
        f.write('accuracy(train): ' + str(round(train_acc_out, 6)) + '\n')
        f.write('accuracy(test): ' + str(round(test_acc_out, 6)) + '\n')
        f.write('f1(train): ' + str(round(train_f1_out, 6)) + '\n')
        f.write('f1(test): ' + str(round(test_f1_out, 6)))

    num_epochs = 5
    train_input = "data/en.train_40.twocol.oov"
    test_input = "data/en.val_40.twocol.oov"
    activation = "relu"

    train_losses1, train_accuracies1, train_f1s1, test_losses1, test_accuracies1, test_f1s1, train_predictions1, test_predictions1 = main(
        train_input, test_input, 64, 64, num_epochs, activation)
    train_losses2, train_accuracies2, train_f1s2, test_losses2, test_accuracies2, test_f1s2, train_predictions2, test_predictions2 = main(
        train_input, test_input, 128, 128, num_epochs, activation)
    train_losses3, train_accuracies3, train_f1s3, test_losses3, test_accuracies3, test_f1s3, train_predictions3, test_predictions3 = main(
        train_input, test_input, 512, 512, num_epochs, activation)

    epochs = range(1, num_epochs + 1)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Plot for embedding and hidden dimension of 64
    plt.plot(epochs, train_f1s1, label='Train F1 (Dim 64)', marker='o', linestyle='-', color='red')
    plt.plot(epochs, test_f1s1, label='Validation F1 (Dim 64)', marker='x', linestyle='--', color='red')

    # Plot for embedding and hidden dimension of 128
    plt.plot(epochs, train_f1s2, label='Train F1 (Dim 128)', marker='o', linestyle='-', color='green')
    plt.plot(epochs, test_f1s2, label='Validation F1 (Dim 128)', marker='x', linestyle='--', color='green')

    # Plot for embedding and hidden dimension of 512
    plt.plot(epochs, train_f1s3, label='Train F1 (Dim 512)', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, test_f1s3, label='Validation F1 (Dim 512)', marker='x', linestyle='--', color='blue')

    # Title and labels
    plt.title('Train and Validation F1 Scores for Different Dimensions')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    plt.show()
