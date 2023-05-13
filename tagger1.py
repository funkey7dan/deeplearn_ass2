import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset

idx_to_label = {}
idx_to_word = {}


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
        hidden_size = 150
        window_size = 5
        input_size = (
            embedding_matrix.embedding_dim * window_size
        )  # 5 concat. 50 dimensional embedding vectors, output over labels
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, output_size)
        # self.softmax = nn.Softmax()
        self.embedding_matrix = embedding_matrix
        self.activate = nn.Tanh()

    def forward(self, x, windows):
        
        # Embeds each word index in a batch of sentences into a dense vector representation using the embedding matrix, and concatenates the resulting embeddings along 
        # the second dimension to create a tensor of shape (batch_size, seq_len * embedding_dim).
        x = torch.cat([self.embedding_matrix(x[:, i]) for i in range(x.shape[1])], dim=1)
        x = self.in_linear(x)
        x = self.activate(x)
        x = self.out_linear(x)
        # x = self.softmax(x) #TODO: we are using cross-entropy loss therefore we maybe don't need softmax
        return x


def train_model(
    model, input_data, dev_data, windows, epochs=1, lr=0.001, input_data_win_index=None
):
    global idx_to_label
    BATCH_SIZE = 32
    # optimizer = torch.optim.SGD(model.parameters(), lr)  # TODO: maybe change to Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # sched = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()
    dropout = nn.Dropout(p=0)

    for j in range(epochs):
        train_loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True)
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            optimizer.zero_grad(set_to_none=True)
            y_hat = model.forward(x, windows)
            y_hat = dropout(y_hat)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        dev_loader = DataLoader(dev_data, batch_size=BATCH_SIZE, shuffle=True)
        running_val_loss = 0
        # Evalaute model on dev at the end of each epoch.
        with torch.no_grad():
            count = 0
            count_no_o = 0
            to_remove = 0
            for k, data in enumerate(dev_loader, 0):
                x, y = data
                y_hat = model.forward(x, windows)
                y_hat = dropout(y_hat)
                val_loss = F.cross_entropy(y_hat, y)
                # Create a list of predicted labels and actual labels
                y_hat_labels = [idx_to_label[i.item()] for i in y_hat.argmax(dim=1)]
                y_labels = [idx_to_label[i.item()] for i in y]
                # Count the number of correct labels th
                y_agreed = sum(
                    [
                        1 if (i == j and j != "o") else 0
                        for i, j in zip(y_hat_labels, y_labels)
                    ]
                )
                count += sum(y_hat.argmax(dim=1) == y).item()
                count_no_o += y_agreed
                to_remove += y_labels.count("o")
                running_val_loss += val_loss.item()
        print(
            f"Epoch {j}, Loss: {train_loss/i}, Dev Loss: {running_val_loss/k}, Dev Acc: {count/(k*BATCH_SIZE)} Acc No O:{count_no_o/((k*BATCH_SIZE)-to_remove)}"
        )


def test_model(model, input_data, windows):
    model.eval()
    total_loss = 0
    for i, data in enumerate(input_data, 0):
        x, y = data
        y_hat = model.forward(x, windows)
        loss = loss_fn(y_hat, y)
        total_loss += loss.item()


def replace_rare(dataset):
    from collections import Counter

    # Define a threshold for word frequency
    threshold = 2

    # Load the dataset into a list of strings (one string per document)

    # Tokenize the dataset into a list of words
    words = [word for doc in dataset for word in doc.split()]

    # Count the frequency of each word
    word_counts = Counter(words)

    # Find the set of rare words (words that occur less than the threshold)
    rare_words = set(word for word in word_counts if word_counts[word] < threshold)

    # Print the rare words
    # print(rare_words)
    updated = [word if word not in rare_words else "<UNK>" for word in dataset]
    return updated


def read_data(fname, window_size=2):

    global idx_to_label
    global idx_to_word
    data = []           
    
    with open(fname) as f:
        lines = f.readlines()
        #lines = [line.strip() for line in lines if line.strip()]
        tokens = []
        labels = []
        sentences = []
        for line in lines:
            if line == "\n":
                sentences.append((tokens, labels))
                tokens = []
                labels = []
                continue
            token, label = line.split("\t")
            tokens.append(token)
            labels.append(label)
    # Preprocess data
    sentences = sentences[1:] # remove docstart
    all_tokens = []
    all_labels = []
    for sentence in sentences:
        tokens, labels = sentence
        for i in range(len(tokens)):
            if any(char.isdigit() for char in tokens[i]) and labels[i] == "O":
                tokens[i] = "$NUM"
            tokens[i] = tokens[i].lower().strip()
            labels[i] = labels[i].lower().strip()
        #tokens = replace_rare(tokens)
        all_tokens.extend(tokens)
        all_labels.extend(labels)
        tokens = np.array(tokens)
        labels = np.array(labels)
        sentence[0].clear()
        sentence[0].extend(tokens)
        sentence[1].clear()
        sentence[1].extend(labels)
    vocab = set(all_tokens) # build a vocabulary of unique tokens
    vocab.add("<PAD>")
    labels_vocab = set(all_labels)

    # Map words to their corresponding index in the vocabulary (word:idx)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    labels_to_idx = {word: i for i, word in enumerate(labels_vocab)}

    idx_to_label = {i: label for label, i in labels_to_idx.items()}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)
    windows = []
    windows_dict = {}
    tokens_idx_all = []
    labels_idx_all = []
    for sentence in sentences:
        tokens, labels = sentence
        # map tokens to their index in the vocabulary
        tokens_idx = [word_to_idx[word] for word in tokens]
        tokens_idx_all.extend(tokens_idx)
        labels_idx = [labels_to_idx[label] for label in labels]
        labels_idx_all.extend(labels_idx)
        
        for i in range(len(tokens_idx)):
            start = max(0, i - window_size)
            end = min(len(tokens_idx), i + window_size + 1)
            context = (
                tokens_idx[start:i]
                + [word_to_idx["<PAD>"]] * (window_size - i + start)
                + tokens_idx[i:end]
                + [word_to_idx["<PAD>"]] * (window_size - end + i + 1)
            )
            label = labels_idx[i]
            windows.append((context, label))
            #windows_dict[i] = (context, label)
    tokens_idx = torch.tensor(tokens_idx_all)
    labels_idx = torch.tensor(labels_idx_all)
    return tokens_idx, labels_idx, windows, vocab, labels_vocab, windows_dict


def idx_to_window_torch(idx, windows, embedding_matrix):
    embedding_size = embedding_matrix.embedding_dim
    batch_size = idx.shape[0]

    # Map the idx_to_window() function to each index in the tensor using PyTorch's map() function
    windows = torch.tensor(list(map(lambda x: windows[x][0], idx.tolist())))
    window_size = windows.size()[1]
    # Use torch.reshape to flatten the tensor of windows into a 2D tensor
    windows_flat = torch.reshape(windows, (batch_size, -1))

    # Index into the embedding matrix to get the embeddings for each word in each window
    # Add a check to ensure that the input word index is within the bounds of the embedding matrix
    embeddings = []
    for i in range(batch_size):
        window = windows_flat[i]
        window_embeddings = []
        for j in range(window_size):
            word_idx = window[j].item()
            if word_idx >= embedding_matrix.num_embeddings or word_idx == -1:
                # If the word index is out of bounds, use the zero vector as the embedding
                embed = torch.zeros((embedding_size,))
            else:
                embed = embedding_matrix(torch.tensor(word_idx))
            window_embeddings.append(embed)
        embeddings.append(torch.cat(window_embeddings))

    # Use torch.stack to stack the tensor of embeddings into a 2D tensor
    embeddings = torch.stack(embeddings)

    return embeddings


def main():
    tokens_idx, labels_idx, windows, vocab, labels_vocab, windows_dict = read_data(
        "./ner/train"
    )
    # embedding_matrix = np.zeros((len(vocab), 50)) #create an empty embedding matrix, each vector is size 50
    embedding_matrix = nn.Embedding(len(vocab), 50, _freeze=False)
    embedding_matrix.weight.requires_grad = True
    nn.init.xavier_uniform_(embedding_matrix.weight)
    model = Tagger("ner", vocab, labels_vocab, embedding_matrix)
    tokenx_idx_new = torch.tensor([window for window, label in windows])
    dataset = TensorDataset(tokenx_idx_new, labels_idx)
    (
        tokens_idx_dev,
        labels_idx_dev,
        windows_dev,
        vocab,
        labels_vocab,
        windows_dict,
    ) = read_data("./ner/dev")
    tokenx_idx_dev_new = torch.tensor([window for window, label in windows_dev])
    dev_dataset = TensorDataset(tokenx_idx_dev_new, labels_idx_dev)
    train_model(
        model, input_data=dataset, dev_data=dev_dataset, epochs=10, windows=windows
    )

    # tokens_idx, labels_idx, windows, vocab, labels_vocab = read_data("./ner/test")


if __name__ == "__main__":
    main()
