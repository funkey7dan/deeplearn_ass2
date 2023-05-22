import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import re
import os
import string
import math
from tagger1 import plot_results

idx_to_label = {}
idx_to_word = {}
word_chars_cache = {}


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
        # self.conv1d = nn.Conv1d(
        #     conv_input_dim,
        #     num_filters,
        #     window_size,
        # )
        self.conv1d = nn.Conv2d(
            conv_input_dim, num_filters, window_size, self.max_word_len
        )
        # TODO: check that the output size is correct
        # self.max_pool = nn.MaxPool1d(kernel_size=max_word_len - window_size + 1)
        # self.max_pool = nn.MaxPool2d(kernel_size=(1, max_word_len - window_size + 1))
        self.char_to_idx = char_to_idx

    # def forward(self, sequence):
    #     # Sequence is a tensor of shape (batch_size, window_size)
    #     global word_chars_cache
    #     batch_size = sequence.shape[0]
    #     # We get a batch of words, each word is a sequence of chars,
    #     # we need to get the char embeddings for each char in each word

    #     # Build a matrix of char embeddings for each word in the batch:
    #     # TODO: make it more efficient with batching
    #     word_matrices = []
    #     padding_tensor = self.char_embedding(torch.tensor(self.char_to_idx["pad"]))

    #     for i in range(batch_size):
    #         # for each word in the window of 5, get the char embeddings
    #         window_torch = []
    #         for j in range(len(sequence[i])):
    #             word = sequence[i][j].item()
    #             word = idx_to_word[word]
    #             word = word.lower()
    #             padding_front = math.floor((self.max_word_len - len(word)) / 2)
    #             padding_back = self.max_word_len - len(word) - padding_front
    #             # Repeat the padding tensor {padding} times
    #             # padding_front_tensor = padding_tensor.repeat(padding_front, 1)
    #             # padding_back_tensor = padding_tensor.repeat(padding_back, 1)

    #             if not word in word_chars_cache:
    #                 word_chars_cache[word] = [self.char_to_idx[char] for char in word]
    #             word = word_chars_cache[word]
    #             word = [self.char_embedding(torch.tensor(char)) for char in word]
    #             word = torch.stack(word)

    #             word_matrix = nn.functional.pad(word, (0, 0, padding_front, padding_back))
    #             # word_matrix = torch.cat(
    #             #     (padding_front_tensor, word, padding_back_tensor), dim=0
    #             # )
    #             window_torch.append(word_matrix)
    #         word_matrices.append(torch.stack(window_torch))
    #     # x = torch.flatten(input=torch.stack(word_matrices), start_dim=1)
    #     x = torch.stack(word_matrices)

    #     # conv1d expects the input to be of shape (batch_size, channels, seq_len)
    #     # so we get a tensor of shape (batch_size, seq_len, channels) and then permute it
    #     #x = x.permute(0, 2, 1)
    #     #TODO: match the dimenions of x to the dimensions of the conv1d, x is of shape torch.Size([1024, 5, 61, 30]s)
    #     x = x.permute(0, 3, 2, 1)
    #     x = self.conv1d(x)
    #     #x = x.squeeze()
    #     x = torch.max(x, dim=2)[0]
    #     #x = self.max_pool(x)
    #     return x.squeeze()

    def forward(self, sequence):
        global word_chars_cache
        # Sequence is a tensor of shape (batch_size, window_size)
        batch_size, window_size = sequence.size()
        max_word_len = self.max_word_len

        # Convert the char_to_idx dictionary into a tensor
        char_to_idx_tensor = torch.tensor(list(self.char_to_idx.values()))

        # Reshape the input sequence to (batch_size * window_size)
        sequence = sequence.view(-1)

        # Convert the sequence of word indices to words
        words = [idx_to_word[idx.item()].lower() for idx in sequence]

        # Convert words to character indices
        char_idx = []
        for word in words:
            if word not in word_chars_cache:
                word_chars_cache[word] = [self.char_to_idx[char] for char in word]
            char_idx.append(word_chars_cache[word])
        [[self.char_to_idx[char] for char in word] for word in words]
        # Pad or truncate the character indices to match max_word_len
        # TODO: change so the padding is done in the beginning and the end of the word
        # char_indices = [
        #     indices[:max_word_len]
        #     if len(indices) > max_word_len
        #     else indices + [0] * (max_word_len - len(indices))
        #     for indices in char_indices
        # ]

        char_indices = [
            [self.char_to_idx["pad"]] * (math.floor((max_word_len - len(word)) / 2))
            + word[:max_word_len]
            + [self.char_to_idx["pad"]]
            * (max_word_len - len(word) - math.floor((max_word_len - len(word)) / 2))
            for word in char_idx
        ]

        # Convert char_indices to a tensor
        char_indices = torch.tensor(char_indices)

        # Reshape char_indices to (batch_size, window_size, max_word_len)
        char_indices = char_indices.view(batch_size, window_size, max_word_len)

        # Apply character embedding
        x = self.char_embedding(char_indices)

        # Reshape x to match the dimensions expected by Conv1d (batch_size * window_size, embedding_size, max_word_len)
        # x = x.view(-1, x.size(2), x.size(3)).transpose(1, 2)
        x = x.permute(0, 3, 2, 1)
        # Apply Conv1d and max-pooling
        x = self.conv1d(x)
        x = torch.max(x, dim=2)[0]
        return x.squeeze()
        # x = self.max_pool(x).squeeze(dim=2)

        # Reshape x to match the original batch size and window size
        x = x.view(batch_size, window_size, -1)

        return x.squeeze(dim=2)


class Tagger(nn.Module):
    def __init__(
        self,
        vocab,
        labels_vocab,
        embedding_matrix,
        char_embedding,
        max_word_len,
        char_to_idx,
    ):
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
        # input_size = embedding_matrix.embedding_dim * window_size
        input_size = (
            char_embedding.embedding_dim + embedding_matrix.embedding_dim * window_size
        )
        hidden_size = 500
        output_size = len(labels_vocab)

        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, output_size)
        self.embedding_matrix = embedding_matrix
        self.char_embedding = char_embedding
        self.word_cnn = WordCNN(
            char_embedding,
            num_filters=30,
            window_size=5,
            max_word_len=max_word_len,
            char_to_idx=char_to_idx,
        )
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x, word_char_embeddings):
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

        x = self.embedding_matrix(x).view(-1, 250)
        x = torch.cat([x, word_char_embeddings], dim=1)
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
            word_char_embeddings = model.word_cnn.forward(x)
            y_hat = model.forward(x, word_char_embeddings)
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
            word_char_embeddings = model.word_cnn.forward(x)
            y_hat = model.forward(x, word_char_embeddings)
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
            word_char_embeddings = model.word_cnn.forward(x)
            y_hat = model.forward(x, word_char_embeddings)
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


def load_embedding_matrix():
    with open("vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]
    vecs = np.loadtxt("wordVectors.txt")
    vecs = torch.from_numpy(vecs)
    vecs = vecs.float()
    return vocab, vecs


def read_data(
    fname,
    embedding_vecs,
    window_size=2,
    vocab=None,
    labels_vocab=None,
    type="train",
    task=None,
    pretrained_vocab=None,
):
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
                token = token.strip().lower() if type == "dev" else token.strip()
                label = label.strip()
                all_labels.append(label)

            else:
                token = line.strip()
                label = ""

            if any(char.isdigit() for char in token) and label == "O":
                token = "NNNUMMM"
            all_tokens.append(token)
            tokens.append(token)
            labels.append(label)

    # if in train case
    if not vocab:
        vocab = set(all_tokens)
        vocab.add("PAD")  # add a padding token
        vocab.add("UUUNKKK")  # add an UUUNKKKnown token

        # train_missing_embeddings = [
        #     word for word in vocab if word not in pretrained_vocab
        # ]
        train_missing_embeddings = set(vocab) - set(pretrained_vocab)
        for word in train_missing_embeddings:
            size = embedding_vecs.shape[1]  # Get the size of the vector
            vector = torch.empty(1, size)
            nn.init.xavier_uniform_(vector)  # Apply Xavier uniform initialization
            embedding_vecs = torch.cat(
                (embedding_vecs, vector), dim=0
            )  # add as last row
            pretrained_vocab.append(word)  # add as last word
        # if we're using pretrained embeddings, the vocabulary changes. turning this into a set doesn't change
        # the size, these are already unique words.
        vocab = pretrained_vocab

    # if in train case
    if not labels_vocab:
        labels_vocab = set(all_labels)

    # Create a vocabulary of unique characters for character embeddings
    char_set = string.ascii_lowercase + string.digits + string.punctuation + " "
    char_vocab = set([char for char in char_set])
    char_vocab = char_vocab.union(set([char for word in all_tokens for char in word]))
    char_vocab.add("pad")

    char_to_idx = {char: i for i, char in enumerate(char_vocab)}
    tokens_to_idx = {token: i for i, token in enumerate(vocab)}
    labels_to_idx = {label: i for i, label in enumerate(labels_vocab)}

    labels_idx = [labels_to_idx[label] for label in all_labels]

    idx_to_label = {i: label for i, label in enumerate(labels_vocab)}
    idx_to_word = {i: word for word, i in tokens_to_idx.items()}
    idx_to_char = {i: char for char, i in char_to_idx.items()}

    windows = []
    char_idx_all = {}
    token_idx_all = []

    for sentence in sentences:
        tokens, labels = sentence
        token_idx_all.extend(
            tokens_to_idx[token.lower()]
            if token.lower() in tokens_to_idx.keys()
            else tokens_to_idx["UUUNKKK"]
            for token in tokens
        )
        for i in range(len(tokens)):
            window = []
            if i < window_size:
                # for _ in range(window_size - i):
                window.extend([tokens_to_idx["PAD"]] * (window_size - i))
            extra_words = tokens[
                max(0, i - window_size) : min(len(tokens), i + window_size + 1)
            ]
            window.extend(
                [
                    tokens_to_idx[token.lower()]
                    if token.lower() in tokens_to_idx.keys()
                    else tokens_to_idx["UUUNKKK"]
                    for token in extra_words
                ]
            )

            if i > len(tokens) - window_size - 1:
                # for _ in range(i - (len(tokens) - window_size - 1)):
                window.extend(
                    [tokens_to_idx["PAD"]] * (i - (len(tokens) - window_size - 1))
                )

            windows.append(window)

    max_len = max([len(word) for word in vocab])
    return (
        torch.tensor(labels_idx),
        torch.tensor(windows),
        vocab,
        labels_vocab,
        labels_to_idx,
        all_tokens,
        embedding_vecs,
        max_len,
        char_to_idx,
        torch.tensor(token_idx_all),
        char_vocab,
    )


def main(task="ner"):
    global idx_to_label
    global idx_to_word
    global word_chars_cache
    idx_to_label = {}
    idx_to_word = {}
    word_chars_cache = {}
    words_embedding_vocabulary, embedding_vecs = load_embedding_matrix()
    (
        labels_idx,
        windows,
        vocab,
        labels_vocab,
        labels_to_idx,
        _,
        embedding_vecs,
        max_word_len,
        char_to_idx,
        token_idx_train,
        char_vocab,
    ) = read_data(
        embedding_vecs=embedding_vecs,
        fname=f"./{task}/train",
        task=task,
        type="train",
        pretrained_vocab=words_embedding_vocabulary,
    )

    (
        labels_idx_dev,
        windows_dev,
        _,
        _,
        _,
        _,
        embedding_vecs,
        max_word_len,
        char_to_idx,
        token_idx_dev,
        char_vocab,
    ) = read_data(
        pretrained_vocab=words_embedding_vocabulary,
        embedding_vecs=embedding_vecs,
        vocab=vocab,
        labels_vocab=labels_vocab,
        fname=f"./{task}/dev",
        task=task,
        type="dev",
    )

    # initialize model and embedding matrix and dataset
    embedding_matrix = nn.Embedding.from_pretrained(embedding_vecs, freeze=False)

    # Initialize the character embedding matrix, each vector is size 30
    chars_embedding = nn.Embedding(len(char_vocab), 30, padding_idx=char_to_idx["pad"])
    nn.init.uniform_(chars_embedding.weight, -math.sqrt(3 / 30), math.sqrt(3 / 30))

    model = Tagger(
        vocab,
        labels_vocab,
        embedding_matrix,
        char_embedding=chars_embedding,
        max_word_len=max_word_len,
        char_to_idx=char_to_idx,
    )

    # the windows were generated according to the new pretrained vocab (plus missing from train) if we use pretrained.
    dataset = TensorDataset(windows, labels_idx)
    dev_dataset = TensorDataset(windows_dev, labels_idx_dev)

    lr_dict = {"ner": 0.0009, "pos": 0.0003}
    epoch_dict = {"ner": 1, "pos": 12}

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
    plot_results(
        dev_loss,
        dev_accuracy,
        dev_accuracy_no_o,
        task,
        tagger_name=(os.path.basename(__file__).split(".")[0]),
    )

    print("Test")
    (
        _,
        windows_test,
        _,
        _,
        _,
        original_tokens,
        embedding_vecs,
        max_word_len,
        char_to_idx,
        token_idx_dev,
        char_vocab,
    ) = read_data(
        embedding_vecs=embedding_vecs,
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
