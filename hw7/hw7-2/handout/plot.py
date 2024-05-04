from rnn import main

import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Define the parameters for the experiments
embedding_dim = 128
hidden_dim = 128
num_epochs = 15
train_input = "data/en.train_40.twocol.oov"
test_input = "data/en.val_40.twocol.oov"

train_losses1, train_accuracies1, train_f1s1, test_losses1, test_accuracies1, test_f1s1, train_predictions1, test_predictions1 = main(
    train_input, test_input, embedding_dim, hidden_dim, num_epochs, "relu")
train_losses2, train_accuracies2, train_f1s2, test_losses2, test_accuracies2, test_f1s2, train_predictions2, test_predictions2 = main(
    train_input, test_input, embedding_dim, hidden_dim, num_epochs, "tanh")

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(10, 6))

plt.plot(epochs, train_f1s1, label='Train F1 (ReLU)', marker='o', linestyle='-', color='red')
plt.plot(epochs, test_f1s1, label='Validation F1 (ReLU)', marker='x', linestyle='--', color='red')

plt.plot(epochs, train_f1s2, label='Train F1 (Tanh)', marker='o', linestyle='-', color='blue')
plt.plot(epochs, test_f1s2, label='Validation F1 (Tanh)', marker='x', linestyle='--', color='blue')

plt.title('Train and Validation F1 Scores by Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)
plt.show()

