import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import string
import math


idx_to_label = {}
idx_to_word = {}
word_chars_cache = {}


class Tagger(nn.Module):
    def __init__(
        self,
        task,
        labels_vocab,
        embedding_matrix,
        chars_embedding,
        char_to_idx,
        window_size=5,
        max_word_len=20,
    ):
        """
        Initializes a Tagger object.

        Args:
            task (str): The task to perform with the model (e.g., "ner" or "pos").
            vocab (set): The vocabulary object for the input data.
            labels_vocab (set): The vocabulary object for the output labels.
            embedding_matrix (torch.nn.Embedding): The matrix of pre-trained embeddings.

        Returns:
            None
        """

        super(Tagger, self).__init__()
        if task == "ner":
            output_size = 5
        else:
            output_size = 36  # assuming https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

        output_size = len(labels_vocab)  # TODO check if it works for us
        hidden_size = 150
        window_size = 5
        # input_size = (
        #     embedding_matrix.embedding_dim * window_size
        # )  # 5 concat. 50 dimensional embedding vectors, output over labels
        input_size = chars_embedding.embedding_dim + embedding_matrix.embedding_dim
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax()
        self.embedding_matrix = embedding_matrix
        self.activate = nn.Tanh()

        # WordCNN parameters
        self.char_to_idx = char_to_idx
        self.char_embedding = char_embedding
        self.word_cnn = WordCNN(
            char_embedding,
            num_filters=30,
            window_size=5,
            max_word_len=max_word_len,
            char_to_idx=char_to_idx,
        )

    def forward(self, x, word_char_embeddings):
        """
        Forward pass of the tagger.
        Expects x of shape (batch_size, window total size), which means 32 windows for example.

        Args:
            x: A tensor of shape (batch_size, seq_len) of word indices.

        Returns:
            A tensor of shape (batch_size, output_dim).
        """
        # Pass the batch to the word embedding layer to get the word embeddings for each word in the batch
        # TODO: check if it works as intended
        # word_char_embeddings = self.word_cnn.forward(x)
        # cnn output shape: (batch_size, conv_input_dim * num_filters, max_word_len)
        x = self.embedding_matrix(x).view(-1, 50)

        x = torch.cat([x, word_char_embeddings], dim=1)
        x = self.in_linear(x)
        x = self.activate(x)
        x = self.out_linear(x)
        return x


class WordCNN(nn.Module):
    def __init__(
        self, char_embedding, num_filters, window_size, max_word_len, char_to_idx
    ):
        super(WordCNN, self).__init__()
        # Get the matrix of char embeddings
        self.char_embedding = char_embedding
        self.char_embedding_dim = char_embedding.embedding_dim
        self.max_word_len = max_word_len

        # The input is the size of each char embedding
        conv_input_dim = self.char_embedding_dim
        self.conv1d = nn.Conv1d(
            conv_input_dim,
            num_filters,
            window_size,
        )
        # TODO: check that the output size is correct
        self.max_pool = nn.MaxPool1d(kernel_size=max_word_len - window_size + 1)
        self.char_to_idx = char_to_idx

    def forward(self, sequence):
        global word_chars_cache
        batch_size = sequence.shape[0]
        # We get a batch of words, each word is a sequence of chars,
        # we need to get the char embeddings for each char in each word

        # Build a matrix of char embeddings for each word in the batch:
        # TODO: make it more efficient with batching
        word_matrices = []
        padding_tensor = self.char_embedding(torch.tensor(self.char_to_idx["pad"]))

        for i in range(batch_size):
            word = sequence[i].item()
            word = idx_to_word[word]
            padding_front = math.floor((self.max_word_len - len(word)) / 2)
            padding_back = self.max_word_len - len(word) - padding_front
            # Repeat the padding tensor {padding} times
            # padding_front_tensor = padding_tensor.repeat(padding_front, 1)
            # padding_back_tensor = padding_tensor.repeat(padding_back, 1)

            if not word in word_chars_cache:
                word_chars_cache[word] = [self.char_to_idx[char] for char in word]
            word = word_chars_cache[word]
            word = [self.char_embedding(torch.tensor(char)) for char in word]
            word = torch.stack(word)

            word_matrix = nn.functional.pad(word, (0, 0, padding_front, padding_back))
            # word_matrix = torch.cat(
            #     (padding_front_tensor, word, padding_back_tensor), dim=0
            # )
            word_matrices.append(word_matrix)
        # x = torch.flatten(input=torch.stack(word_matrices), start_dim=1)
        x = torch.stack(word_matrices)

        # conv1d expects the input to be of shape (batch_size, channels, seq_len)
        # so we get a tensor of shape (batch_size, seq_len, channels) and then permute it
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.max_pool(x)
        return x.squeeze()


def train_model(
    model, input_data, dev_data, windows, epochs=1, lr=0.01, input_data_win_index=None
):
    """
    Trains a given model using the provided input and development data.
    Args:
            model: The model to train.
            input_data: The training data.
            dev_data: The development data to evaluate the model.
            windows: The size of the windows to use.
            epochs: The number of epochs to train the model for. Default value is 1.
            lr: The learning rate to use. Default value is 0.01.
            input_data_win_index: The index of the window in the input data. Default value is None.
    Returns:
            A list of the accuracy of the model on the development data at the end of each epoch.
    """

    global idx_to_label
    BATCH_SIZE = 256
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    dropout = nn.Dropout(p=0.5)
    dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, windows)
    print(
        f"Before Training, Dev Loss: {dev_loss}, Dev Acc: {dev_acc} Acc No O:{dev_acc_clean}"
    )

    best_loss = 100000
    best_weights = None
    dev_loss_results = []
    dev_acc_results = []
    dev_acc_no_o_results = []

    for j in range(epochs):
        model.train()
        train_loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            optimizer.zero_grad(set_to_none=True)
            word_char_embeddings = model.word_cnn.forward(x)
            y_hat = model.forward(x, word_char_embeddings)
            y_hat = dropout(y_hat)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # Evaluate model on dev at the end of each epoch.
        dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, windows)

        # Save best model
        if dev_loss < best_loss:
            best_loss = dev_loss
            best_weights = model.state_dict()

        if task == "ner":
            print(
                f"Epoch {j+1}/{epochs}, Loss: {train_loss/i}, Dev Loss: {dev_loss}, Dev Acc: {dev_acc} Acc No O:{dev_acc_clean}"
            )
        else:
            print(
                f"Epoch {j+1}/{epochs}, Loss: {train_loss/i}, Dev Loss: {dev_loss}, Dev Acc: {dev_acc}"
            )
        sched.step()
        dev_loss_results.append(dev_loss)
        dev_acc_results.append(dev_acc)
        dev_acc_no_o_results.append(dev_acc_clean)
    # load best weights
    model.load_state_dict(best_weights)
    filename = os.path.basename(__file__).split(".")[0]
    torch.save(model, f"{filename}_best_model_{task}.pth")
    return dev_loss_results, dev_acc_results, dev_acc_no_o_results


def test_model(model, input_data, windows):
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
            y_hat = model.forward(x, model.word_cnn.forward(x))
            # y_hat = dropout(y_hat)
            val_loss = F.cross_entropy(y_hat, y)
            # Create a list of predicted labels and actual labels
            y_hat_labels = [idx_to_label[i.item()] for i in y_hat.argmax(dim=1)]
            y_labels = [idx_to_label[i.item()] for i in y]
            # Count the number of correct labels th
            y_agreed = sum(
                [
                    1 if (i == j and j != "O") else 0
                    for i, j in zip(y_hat_labels, y_labels)
                ]
            )
            count += sum(y_hat.argmax(dim=1) == y).item()
            count_no_o += y_agreed
            to_remove += y_labels.count("O")
            running_val_loss += val_loss.item()

    return (
        running_val_loss / k,
        count / (k * BATCH_SIZE),
        count_no_o / ((k * BATCH_SIZE) - to_remove),
    )


def replace_rare(dataset, threshold=1):
    from collections import Counter

    # Count the frequency of each word
    word_counts = Counter(dataset)

    # Find the set of rare words (words that occur less than the threshold)
    rare_words = set(word for word in word_counts if word_counts[word] < threshold)

    # Print the rare words
    # print(rare_words)
    updated = [word if word not in rare_words else "UUUNKKK" for word in dataset]
    return updated


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
    data = []
    SEPARATOR = "\t" if task == "ner" else " "

    with open(fname) as f:
        lines = f.readlines()
        # lines = [line.strip() for line in lines if line.strip()]
        tokens = []
        labels = []
        sentences = []
        for line in lines:
            if line == "\n":
                sentences.append((tokens, labels))
                tokens = []
                labels = []
                continue
            if type != "test":
                token, label = line.split(SEPARATOR)
            else:
                token = line.strip()
                label = ""
            if any(char.isdigit() for char in token) and label == "O":
                token = "NNNUMMM"
            tokens.append(token)
            labels.append(label)
    # Preprocess data
    sentences = sentences[1:]  # remove docstart
    all_tokens = []
    all_labels = []
    for sentence in sentences:
        tokens, labels = sentence
        for i in range(len(tokens)):
            tokens[i] = tokens[i].strip().lower()
            labels[i] = labels[i].strip()

        all_tokens.extend(tokens)
        all_labels.extend(labels)
        tokens = np.array(tokens)
        labels = np.array(labels)
        sentence[0].clear()
        sentence[0].extend(tokens)
        sentence[1].clear()
        sentence[1].extend(labels)
    # tokens = replace_rare(tokens)
    all_tokens.extend(["<PAD>", "UUUNKKK"])
    if not vocab:
        # all_tokens.extend(["<PAD>","UUUNKKK"])
        vocab = set(all_tokens)  # build a vocabulary of unique tokens
    # vocab.add("<PAD>")  # add a padding token
    # vocab.add("UUUNKKK")  # add an unknown token
    if not labels_vocab:
        labels_vocab = set(all_labels)

    # Create a vocabulary of unique characters for character embeddings
    char_set = string.ascii_lowercase + string.digits + string.punctuation + " "
    char_vocab = set([char for char in char_set])
    char_vocab = char_vocab.union(set([char for word in all_tokens for char in word]))
    char_vocab.add("pad")
    char_to_idx = {char: i for i, char in enumerate(char_vocab)}

    # Map words to their corresponding index in the vocabulary (word:idx)
    word_to_idx = {word: i for i, word in enumerate(vocab)}

    labels_to_idx = {word: i for i, word in enumerate(labels_vocab)}

    idx_to_label = {i: label for label, i in labels_to_idx.items()}
    idx_to_word = {i: word for word, i in word_to_idx.items()}
    idx_to_char = {i: char for char, i in char_to_idx.items()}

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)

    windows = []

    windows_dict = {}
    tokens_idx_all = []
    labels_idx_all = []
    max_len = 0

    for sentence in sentences:
        tokens, labels = sentence

        # map tokens to their index in the vocabulary
        tokens_idx = [
            word_to_idx[word] if word in word_to_idx else word_to_idx["UUUNKKK"]
            for word in tokens
        ]

        # find the maximum length of a token in the data
        max_len = max(max_len, max(len(word) for word in tokens))
        tokens_idx_all.extend(tokens_idx)

        labels_idx = [
            labels_to_idx[label] if label in labels_to_idx else labels_to_idx["O"]
            for label in labels
        ]
        labels_idx_all.extend(labels_idx)

        for i in range(len(tokens_idx)):
            start = max(0, i - window_size)
            end = min(len(tokens_idx), i + window_size + 1)
            context = (
                [word_to_idx["<PAD>"]] * (window_size - i + start)
                + tokens_idx[start:i]
                + tokens_idx[i:end]
                + [word_to_idx["<PAD>"]] * (window_size - end + i + 1)
            )
            label = labels_idx[i]
            windows.append((context, label))

    tokens_idx = torch.tensor(tokens_idx_all)
    labels_idx = torch.tensor(labels_idx_all)

    return (
        tokens_idx,
        labels_idx,
        windows,
        vocab,
        labels_vocab,
        windows_dict,
        char_vocab,
        char_to_idx,
        max_len,
    )


def load_embedding_matrix(embedding_path):
    with open("vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]
        vocab = set(vocab)
    vecs = np.loadtxt("wordVectors.txt")
    vecs = torch.from_numpy(vecs)
    vecs = vecs.float()
    return vocab, vecs


def plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task):
    # # Plot the dev loss, and save
    tagger_name = os.path.basename(__file__).split(".")[0]
    plt.plot(dev_loss, label="dev loss")
    plt.title(f"{task} task")
    plt.savefig(f"loss_{task}_{tagger_name}.png")
    # plt.show()
    #
    # # Plot the dev accuracy, and save
    plt.plot(dev_accuracy, label="dev accuracy")
    plt.title(f"{task} task")
    plt.savefig(f"accuracy_{task}_{tagger_name}.png")
    # plt.show()
    #
    # # Plot the dev accuracy no O, and save
    plt.plot(dev_accuracy_no_o, label="dev accuracy no o")
    plt.title(f"{task} task")
    plt.savefig(f"accuracy_no_O_{task}_{tagger_name}.png")
    # plt.show()


def main(task="ner"):
    _, vecs = load_embedding_matrix(f"./wordVectors.txt")
    (
        tokens_idx,
        labels_idx,
        windows,
        vocab,
        labels_vocab,
        windows_dict,
        char_vocab,
        char_to_idx,
        max_len,
    ) = read_data(f"./{task}/train", task=task)
    # Create embedding matrices for the prefixes and suffixes

    dataset = TensorDataset(tokens_idx, labels_idx)

    # Initialize the character embedding matrix, each vector is size 30
    chars_embedding = nn.Embedding(len(char_vocab), 30, padding_idx=char_to_idx["pad"])
    nn.init.uniform_(chars_embedding.weight, -math.sqrt(3 / 30), math.sqrt(3 / 30))

    # create an empty embedding matrix, each vector is size 50
    # embedding_matrix = nn.Embedding(len(vocab), 50, _freeze=False)
    # initialize the embedding matrix to random values using xavier initialization which is a good initialization for NLP tasks
    # nn.init.xavier_uniform_(embedding_matrix.weight)

    embedding_matrix = nn.Embedding.from_pretrained(
        vecs,
        freeze=False,
    )
    # embedding_matrix.weight.requires_grad = True

    model = Tagger(
        task,
        vocab,
        labels_vocab,
        embedding_matrix,
        chars_embedding = chars_embedding,
        window_size=1,
        max_word_len=max_len,
        char_to_idx=char_to_idx,
    )

    # Load the dev data
    (
        tokens_idx_dev,
        labels_idx_dev,
        windows_dev,
        vocab,
        labels_vocab,
        windows_dict,
        char_vocab,
        char_to_idx,
        max_len,
    ) = read_data(f"./{task}/dev", task=task, vocab=vocab, labels_vocab=labels_vocab)
    tokens_idx_dev_new = torch.tensor([window for window, label in windows_dev])

    dev_dataset = TensorDataset(tokens_idx_dev, labels_idx_dev)
    # Get the dev loss from the model training
    dev_loss, dev_accuracy, dev_accuracy_no_o = train_model(
        model, input_data=dataset, dev_data=dev_dataset, epochs=5, windows=windows
    )

    torch.save(model, f"model_{task}.pt")
    plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task=task)


if __name__ == "__main__":
    tasks = ["ner", "pos"]
    for task in tasks:
        main(task)
