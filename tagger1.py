import torch
import torch.nn as nn
import numpy as np
import os


class Tagger(nn.Module):
    # TODO clean this description
    """
    A sequence tagger,where
    the input is a sequence of items(in our case, a sentence of natural-language words),
    and an output is a label for each of the item.
    The tagger will be greedy/local and window-based. For a sequence of words
    w1,...,wn, the tag of word wi will be based on the words in a window of
    two words to each side of the word being tagged: wi-2,wi-1,wi,wi+1,wi+2.
    'Greedy/local' here means that the tag assignment for word i will not depend on the tags of other words
    each word in the window will be assigned a 50 dimensional embedding vector, using an embedding matrix E.
    MLP with one hidden layer and tanh activation function.
    The output of the MLP will be passed through a softmax transformation, resulting in a probability distribution.
    The network will be trained with a cross-entropy loss.
    The vocabulary of E will be based on the words in the training set (you are not allowed to add to E words that appear only in the dev set).
    """

    def __init__(self, task, vocab, labels_vocab, embedding_matrix):
        super(Tagger, self).__init__()
        if task == "ner":
            output_size = 5
        else:
            output_size = 36  # assumming https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        hidden_size = 250
        window_size = 5
        input_size = (
            embedding_matrix.embedding_dim * window_size
        )  # 4 concat. 50 dimensional embedding vectors, output over labels
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax()
        self.embedding_matrix = embedding_matrix
        self.activate = nn.Tanh()

    def forward(self, x, windows):
        # get embedding vector for input
        x = idx_to_window_torch(x, windows, self.embedding_matrix)
        # x = self.embedding_matrix(x)
        x = self.in_linear(x)
        x = self.activate(x)
        x = self.out_linear(x)
        x = self.softmax(x)
        return x


def read_data(fname, window_size=2):
    """
    Read data from a file and return token and label indices, vocabulary, and label vocabulary.

    Args:
        fname (str): The name of the file to read data from.
        window_size (int, optional): The size of the window for the token,from each side of the word. Defaults to 2.

    Returns:
        tuple: A tuple containing:
            - tokens_idx (numpy.ndarray): An array of token indices.
            - labels_idx (numpy.ndarray): An array of label indices.
            - vocab (set): A set of unique tokens in the data.
            - labels_vocab (set): A set of unique labels in the data.
    """
    data = []
    with open(fname) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if line.strip()]
        tokens = []
        labels = []
        for line in lines:
            token, label = line.split()
            tokens.append(token)
            labels.append(label)

    tokens = np.array(tokens[1:])
    labels = np.array(labels[1:])
    vocab = set(tokens)  # build a vocabulary of unique tokens
    labels_vocab = set(labels)

    # Map words to their corresponding index in the vocabulary (word:idx)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    labels_to_idx = {word: i for i, word in enumerate(labels_vocab)}

    # For each window, map tokens to their index in the vocabulary
    tokens_idx = [word_to_idx[word] for word in tokens]

    # tokens_idx = torch.from_numpy(tokens_idx)
    labels_idx = [labels_to_idx[label] for label in labels]
    # labels_idx = torch.from_numpy(labels_idx)

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)
    windows = []
    for i in range(len(tokens_idx)):
        start = max(0, i - window_size)
        end = min(len(tokens_idx), i + window_size + 1)
        context = (
            tokens_idx[start:i]
            + [-1] * (window_size - i + start)
            + tokens_idx[i:end]
            + [-1] * (window_size - end + i + 1)
        )
        label = labels_idx[i]
        windows.append((context, label))

    tokens_idx = torch.tensor(tokens_idx)
    labels_idx = torch.tensor(labels_idx)
    return tokens_idx, labels_idx, windows, vocab, labels_vocab


def idx_to_window_torch(idx, windows, embedding_matrix):
    t_new = torch.tensor([])
    tensors = []
    for t in windows[idx][0]:
        if t != -1:
            t = embedding_matrix(torch.tensor(t))
        else:
            t = torch.zeros(embedding_matrix.embedding_dim)
        tensors.append(t)
    return torch.cat(tensors)


def train_model(model, input_data, windows, epochs=1, lr=0.01):
    optimizer = torch.optim.SGD(model.parameters(), lr)  # TODO: maybe change to Adam
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    for j in range(epochs):
        train_loss = 0
        optimizer.zero_grad()
        for i, data in enumerate(input_data, 0):
            x, y = data
            y_hat = model.forward(x, windows)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch {j}, Loss: {train_loss/i}")

def test_model(model, input_data, windows):
    model.eval()
    total_loss = 0
    for i, data in enumerate(input_data, 0):
        x, y = data
        y_hat = model.forward(x, windows)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()


def main():
    tokens_idx, labels_idx, windows, vocab, labels_vocab = read_data("./ner/train")
    # embedding_matrix = np.zeros((len(vocab), 50)) #create an empty embedding matrix, each vector is size 50
    embedding_matrix = nn.Embedding(len(vocab), 50)
    model = Tagger("ner", vocab, labels_vocab, embedding_matrix)
    train_model(
        model, input_data=zip(tokens_idx, labels_idx), epochs=1, windows=windows
    )
    tokens_idx, labels_idx, windows, vocab, labels_vocab = read_data("./ner/test")

if __name__ == "__main__":
    main()
