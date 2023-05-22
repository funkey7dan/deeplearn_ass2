import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import re
import os

idx_to_label = {}
idx_to_word = {}


class Tagger(nn.Module):
    def __init__(self, vocab, labels_vocab, embedding_matrix):
        """
        Initializes a Tagger object.

        Args:
            vocab (set): The vocabulary object for the input data.
            labels_vocab (set): The vocabulary object for the output labels.
            embedding_matrix (torch.nn.Embedding): The matrix of pre-trained embeddings.

        Returns:
            None
        """
        super(Tagger, self).__init__()
        self.vocab = vocab
        window_size = 5

        # 5 concat. 50 dimensional embedding vectors, output over labels
        input_size = embedding_matrix.embedding_dim * window_size
        hidden_size = 500
        output_size = len(labels_vocab)

        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, output_size)
        self.embedding_matrix = embedding_matrix
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        """
        Forward pass of the tagger.
        Expects x of shape (batch_size, window total size), which means 32 windows of 5 for example.

        Args:
            x: A tensor of shape (batch_size, window total size) of word indices.

        Returns:
            A tensor of shape (batch_size, output_dim).
        """
        # Concatenate the word embedding vectors of the words in the window to create a single vector for the window.
        # x.shape[1] is the size of window, which is 5 in our case.
        # self.embedding_matrix(x[:,i]) is the 32 x 50 embedding matrix of the i'th items in the window.
        # torch.cat concatenates the 5 32 x 50 embedding matrix of the i'th items in the window to a 32 x 250 matrix, which we can pass to the linear layer.
        # x = torch.cat(
        #     [self.embedding_matrix(x[:, i]) for i in range(x.shape[1])], dim=1
        # )
        x = self.embedding_matrix(x).view(
            -1, 250
        )  # Faster than the above and equivalent
        x = self.in_linear(x)
        x = self.tanh(x)
        x = self.dropout(x)
        x = self.out_linear(x)
        return x


def train_model(
    model,
    input_data,
    dev_data,
    labels_to_idx,
    epochs=1,
    lr=0.0001,
    task="ner",
):
    """
    Trains a given model using the provided input and development data.
    Args:
            model: The model to train.
            input_data: The training data.
            dev_data: The development data to evaluate the model.
            epochs: The number of epochs to train the model for. Default value is 1.
            lr: The learning rate to use. Default value is 0.01.
    Returns:
            A list of the accuracy of the model on the development data at the end of each epoch.
    """

    global idx_to_label
    BATCH_SIZE = 1024
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    # sched = torch.optim.lr_scheduler.StepLR(
    #     optimizer, step_size=3, gamma=0.1
    # )  # scheduler that every 3 epochs it updates the lr

    # dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, task=task)
    # print(
    #     f"Before Training, Dev Loss: {dev_loss}, Dev Acc: {dev_acc} Acc No O:{dev_acc_clean}"
    # )

    best_loss = 100000
    best_weights = None
    dev_loss_results = []
    dev_acc_results = []
    dev_acc_no_o_results = []

    for j in range(epochs):
        train_loader = DataLoader(
            input_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            optimizer.zero_grad(set_to_none=True)
            y_hat = model.forward(x)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Evaluate model on dev at the end of each epoch.
        dev_loss, dev_acc, dev_acc_clean = test_model(
            model=model, task=task, labels_to_idx=labels_to_idx, input_data=dev_data
        )

        # Save best model
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_weights = model.state_dict()

        if task == "ner":
            print(
                f"Epoch {j + 1}/{epochs}, Loss: {train_loss / i}, Dev Loss: {dev_loss}, Dev Acc: {dev_acc} Acc No O:{dev_acc_clean}"
            )
        else:
            print(
                f"Epoch {j + 1}/{epochs}, Loss: {train_loss / i}, Dev Loss: {dev_loss}, Dev Acc: {dev_acc}"
            )

        # sched.step()
        dev_loss_results.append(dev_loss)
        dev_acc_results.append(dev_acc)
        dev_acc_no_o_results.append(dev_acc_clean)

    # load best weights
    model.load_state_dict(best_weights)
    filename = os.path.basename(__file__).split(".")[0]
    torch.save(model, f"{filename}_best_model_{task}.pth")
    return dev_loss_results, dev_acc_results, dev_acc_no_o_results


def test_model(model, input_data, task, labels_to_idx):
    """
    This function tests a PyTorch model on given input data and returns the validation loss, overall accuracy, and
    accuracy excluding "O" labels. It takes in the following parameters:

    - model: a PyTorch model to be tested
    - input_data: a dataset to test the model on

    The function first initializes a batch size of 32 and a global variable idx_to_label. It then creates a DataLoader
    object with the input_data and the batch size, and calculates the validation loss, overall accuracy, and accuracy
    excluding "O" labels. These values are returned as a tuple.
    """

    BATCH_SIZE = 1024
    global idx_to_label

    loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)
    running_val_loss = 0

    with torch.no_grad():
        model.eval()
        count = 0
        count_no_o = 0
        to_remove = 0

        for k, data in enumerate(loader, 0):
            x, y = data
            y_hat = model.forward(x)
            val_loss = F.cross_entropy(y_hat, y)

            y_hat_labels = [i.item() for i in y_hat.argmax(dim=1)]
            y_labels = [i.item() for i in y]

            if task == "ner":
                # Count the number of correct labels th
                y_agreed = sum(
                    [
                        1 if (i == j and j != labels_to_idx["O"]) else 0
                        for i, j in zip(y_hat_labels, y_labels)
                    ]
                )
            else:
                y_agreed = sum(
                    [1 if (i == j) else 0 for i, j in zip(y_hat_labels, y_labels)]
                )

            count += sum(y_hat.argmax(dim=1) == y).item()
            count_no_o += y_agreed

            if task == "ner":
                to_remove += y_labels.count(labels_to_idx["O"])

            running_val_loss += val_loss.item()

    return (
        running_val_loss / k,
        count / (k * BATCH_SIZE),
        count_no_o / ((k * BATCH_SIZE) - to_remove),
    )


def run_inference(model, input_data, task, original_tokens):
    """
    This function tests a PyTorch model on given input data and returns the validation loss, overall accuracy, and
    accuracy excluding "O" labels. It takes in the following parameters:

    - model: a PyTorch model to be tested
    - input_data: a dataset to test the model on
    - windows: a parameter that is not used in the function

    The function first initializes a batch size of 32 and a global variable idx_to_label. It then creates a DataLoader
    object with the input_data and the batch size, and calculates the validation loss, overall accuracy, and accuracy
    excluding "O" labels. These values are returned as a tuple.
    """

    BATCH_SIZE = 256
    global idx_to_label, idx_to_word

    loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    predictions = []

    with torch.no_grad():
        model.eval()

        j = 0
        for _, data in enumerate(loader, 0):
            x, y = data
            y_hat = model.forward(x)
            y_hat = model.softmax(y_hat)
            x_words = [original_tokens[i + j] for i, _ in enumerate(x)]
            y_hat_labels = [idx_to_label[i.item()] for i in y_hat.argmax(dim=1)]
            predictions.extend(zip(x_words, y_hat_labels))
            j += BATCH_SIZE

    # find which tagger we are using
    tagger_idx = re.findall(r"\d+", os.path.basename(__file__))[0]
    with open(f"test{tagger_idx}.{task}", "w") as f:
        for pred in predictions:
            f.write(f"{pred[0]} {pred[1]}" + "\n")


def read_data(
    fname, window_size=2, vocab=None, labels_vocab=None, type="train", task=None
):
    """
    Reads in data from a file and preprocesses it for use in a neural network model.

    Args:
        fname (str): The name of the file to read data from.
        window_size (int, optional): The size of the context window to use when creating windows. Defaults to 2.
        vocab (set, optional): A set of unique tokens to use as the vocabulary. If not provided, a vocabulary will be built from the data. Defaults to None.
        labels_vocab (set, optional): A set of unique labels to use for classification. If not provided, labels will be inferred from the data. Defaults to None.
        type (str, optional): A string indicating the type of data being read. Defaults to "train".

    Returns:
        tuple: A tuple containing the preprocessed data:
            - tokens_idx (torch.Tensor): A tensor of indices representing the tokens in the data.
            - labels_idx (torch.Tensor): A tensor of indices representing the labels in the data.
            - windows (list): A list of tuples representing the context windows and labels for each token in the data.
            - vocab (set): A set of unique tokens used as the vocabulary.
            - labels_vocab (set): A set of unique labels used for classification.
            - windows_dict (dict): A dictionary mapping token indices to their corresponding context windows and labels.
            - task (str): The task to perform.
    """

    global idx_to_label
    global idx_to_word

    SEPARATOR = "\t" if task == "ner" else " "

    all_tokens = []
    all_labels = []

    # the sentences always remain the same, regardless of the method used: an index for each word.
    with open(fname) as f:
        lines = f.readlines()
        tokens = []
        labels = []
        sentences = []

        for line in lines:
            if line == "\n":
                sentences.append((np.array(tokens), np.array(labels)))
                tokens = []
                labels = []
                continue

            if type != "test":
                token, label = line.split(SEPARATOR)
                if type == "dev":
                    token = token.strip()
                else:
                    token = token.strip()
                label = label.strip()
                all_labels.append(label)
            else:
                token = line.strip()
                label = ""

            if any(char.isdigit() for char in token) and label == "O":
                token = "NUM"
            all_tokens.append(token)
            tokens.append(token)
            labels.append(label)

    # if in train case
    if not vocab:
        vocab = set(all_tokens)
        vocab.add("PAD")  # add a padding token
        vocab.add("UUUNKKK")  # add an unknown token

    # if in train case
    if not labels_vocab:
        labels_vocab = set(all_labels)

    tokens_to_idx = {token: i for i, token in enumerate(vocab)}
    labels_to_idx = {label: i for i, label in enumerate(labels_vocab)}

    labels_idx = [labels_to_idx[label] for label in all_labels]
    idx_to_label = {i: label for i, label in enumerate(labels_vocab)}
    idx_to_word = {i: word for word, i in tokens_to_idx.items()}
    windows = []

    for sentence in sentences:
        tokens, labels = sentence
        for i in range(len(tokens)):
            window = []
            if i < window_size:
                for _ in range(window_size - i):
                    window.append(tokens_to_idx["PAD"])
            extra_words = tokens[
                max(0, i - window_size) : min(len(tokens), i + window_size + 1)
            ]
            window.extend(
                [
                    tokens_to_idx[token]
                    if token in tokens_to_idx.keys()
                    else tokens_to_idx["UUUNKKK"]
                    for token in extra_words
                ]
            )

            if i > len(tokens) - window_size - 1:
                for _ in range(i - (len(tokens) - window_size - 1)):
                    window.append(tokens_to_idx["PAD"])

            windows.append(window)

    return (
        torch.tensor(labels_idx),
        torch.tensor(windows),
        vocab,
        labels_vocab,
        labels_to_idx,
        all_tokens,
    )


def plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task, tagger_name=None):
    # Plot the dev loss, and save
    if not tagger_name: tagger_name = os.path.basename(__file__).split(".")[0]
    plt.plot(range(1, len(dev_loss) + 1), dev_loss, label="dev loss")
    plt.xticks(range(1, len(dev_loss) + 1))
    plt.title(f"dev loss {task}")
    plt.savefig(f"./images/loss_{task}_{tagger_name}.png")
    plt.clf()

    # Plot the dev accuracy, and save
    plt.plot(range(1, len(dev_accuracy) + 1), dev_accuracy, label="dev accuracy")
    plt.xticks(range(1, len(dev_accuracy) + 1))
    plt.title(f"dev accuracy {task}")
    plt.savefig(f"./images/accuracy_{task}_{tagger_name}.png")
    plt.clf()

    # Plot the dev accuracy no O, and save
    plt.plot(
        range(1, len(dev_accuracy_no_o) + 1),
        dev_accuracy_no_o,
        label="dev accuracy no o",
    )
    plt.xticks(range(1, len(dev_accuracy_no_o) + 1))
    plt.title(f"dev accuracy no 'O' {task}")
    plt.savefig(f"./images/accuracy_no_O_{task}_{tagger_name}.png")
    plt.clf()


def main(task="ner"):
    labels_idx, windows, vocab, labels_vocab, labels_to_idx, _ = read_data(
        fname=f"./{task}/train", task=task, type="train"
    )

    labels_idx_dev, windows_dev, _, _, _, _ = read_data(
        vocab=vocab,
        labels_vocab=labels_vocab,
        fname=f"./{task}/dev",
        task=task,
        type="dev",
    )

    # initialize model and embedding matrix and dataset
    embedding_matrix = nn.Embedding(len(vocab), 50)
    nn.init.xavier_uniform_(embedding_matrix.weight)

    model = Tagger(vocab, labels_vocab, embedding_matrix)

    # the windows were generated according to the new pretrained vocab (plus missing from train) if we use pretrained.
    dataset = TensorDataset(windows, labels_idx)
    dev_dataset = TensorDataset(windows_dev, labels_idx_dev)

    lr_dict = {"ner": 0.0007, "pos": 0.0001}
    epoch_dict = {"ner": 5, "pos": 8}

    # train model
    dev_loss, dev_accuracy, dev_accuracy_no_o = train_model(
        model,
        input_data=dataset,
        dev_data=dev_dataset,
        epochs=epoch_dict[task],
        lr=lr_dict[task],
        labels_to_idx=labels_to_idx,
        task=task,
    )

    # plot the results
    plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task)

    print("Test")
    _, windows_test, _, _, _, original_tokens = read_data(
        fname=f"./{task}/test",
        vocab=vocab,
        labels_vocab=labels_vocab,
        type="test",
        task=task,
    )

    test_dataset = TensorDataset(windows_test, torch.tensor([0] * len(windows_test)))

    run_inference(
        model=model, input_data=test_dataset, task=task, original_tokens=original_tokens
    )


if __name__ == "__main__":
    tasks = ["ner", "pos"]
    for task in tasks:
        print(task)
        main(task)
