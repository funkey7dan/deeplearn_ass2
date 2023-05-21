import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

idx_to_label = {}
idx_to_word = {}


class Tagger(nn.Module):
    def __init__(self, task, vocab, labels_vocab, embedding_matrix):
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
        self.task = task
        self.vocab = vocab
        output_size = len(labels_vocab)
        hidden_size = 150
        window_size = 5
        input_size = (
            embedding_matrix.embedding_dim * window_size
        )  # 5 concat. 50 dimensional embedding vectors, output over labels
        self.in_linear = nn.Linear(input_size, hidden_size)
        self.out_linear = nn.Linear(hidden_size, output_size)
        self.embedding_matrix = embedding_matrix
        self.activate = nn.Tanh()

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
        x = self.activate(x)
        x = self.out_linear(x)
        return x


def train_model(
    model,
    input_data,
    dev_data,
    epochs=1,
    lr=0.01,
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
    BATCH_SIZE = 256
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=3, gamma=0.1
    )  # scheduler that every 3 epochs it updates the lr
    dropout = nn.Dropout(p=0.5)
    dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, task=task)
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
        train_loader = DataLoader(
            input_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
        )
        train_loss = 0
        for i, data in enumerate(train_loader, 0):
            x, y = data
            optimizer.zero_grad(set_to_none=True)
            y_hat = model.forward(x)
            y_hat = dropout(y_hat)
            loss = F.cross_entropy(y_hat, y)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        # Evaluate model on dev at the end of each epoch.
        dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_data, task=task)

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
    filename = os.path.basename(__file__)[0].split(".")[0]
    torch.save(model, f"{filename}_best_model_{task}.pth")
    return dev_loss_results, dev_acc_results, dev_acc_no_o_results


def test_model(model, input_data, task):
    """
    This function tests a PyTorch model on given input data and returns the validation loss, overall accuracy, and
    accuracy excluding "O" labels. It takes in the following parameters:

    - model: a PyTorch model to be tested
    - input_data: a dataset to test the model on

    The function first initializes a batch size of 32 and a global variable idx_to_label. It then creates a DataLoader
    object with the input_data and the batch size, and calculates the validation loss, overall accuracy, and accuracy
    excluding "O" labels. These values are returned as a tuple.
    """

    BATCH_SIZE = 256
    global idx_to_label

    loader = DataLoader(input_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    running_val_loss = 0
    with torch.no_grad():
        model.eval()
        count = 0
        count_no_o = 0
        to_remove = 0
        for k, data in enumerate(loader, 0):
            x, y = data
            y_hat = model.forward(x)
            # y_hat = dropout(y_hat)
            val_loss = F.cross_entropy(y_hat, y)
            # Create a list of predicted labels and actual labels
            count += sum(y_hat.argmax(dim=1) == y).item()
            running_val_loss += val_loss.item()
            if task == "ner":
                y_hat_labels = [idx_to_label[i.item()] for i in y_hat.argmax(dim=1)]
                y_labels = [idx_to_label[i.item()] for i in y]
                # Count the number of correct labels th
                y_agreed = sum(
                    [
                        1 if (i == j and j != "O") else 0
                        for i, j in zip(y_hat_labels, y_labels)
                    ]
                )
                count_no_o += y_agreed
                to_remove += y_labels.count("O")
            else:
                count_no_o = count
                to_remove = 0

    return (
        running_val_loss / k,
        count / (k * BATCH_SIZE),
        count_no_o / ((k * BATCH_SIZE) - to_remove),
    )


def run_inference(model, input_data, task, original_words):
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
        for k, data in enumerate(loader, 0):
            x, y = data
            y_hat = model.forward(x)
            x_tokens = [original_words[i + BATCH_SIZE * k] for i, _ in enumerate(x)]
            y_hat_labels = [idx_to_label[i.item()] for i in y_hat.argmax(dim=1)]
            predictions.extend(zip(x_tokens, y_hat_labels))

    # find which tagger we are using
    tagger_idx = re.findall(r"\d+", os.path.basename(__file__))[0]
    with open(f"test{tagger_idx}.{task}", "w") as f:
        for pred in predictions:
            f.write(f"{pred[0]} {pred[1]}" + "\n")


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

    if task == "ner":
        SEPARATOR = "\t"
    else:
        SEPARATOR = " "

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
    if type != "test":
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
    # TODO Best accuracy with this for threshold 2: ~0.835, without it 0.86+, when reducing to 1, ~0.868
    # all_tokens = replace_rare(all_tokens)
    if not vocab:
        vocab = set(all_tokens)  # build a vocabulary of unique tokens
        vocab.add("<PAD>")  # add a padding token
        vocab.add("UUUNKKK")  # add an unknown token
    if not labels_vocab:
        labels_vocab = set(all_labels)

    # Map words to their corresponding index in the vocabulary (word:idx)
    word_to_idx = {word: i for i, word in enumerate(vocab)}
    labels_to_idx = {word: i for i, word in enumerate(labels_vocab)}

    if type != "test":
        idx_to_label = {i: label for label, i in labels_to_idx.items()}
    idx_to_word = {i: word for word, i in word_to_idx.items()}

    # Create windows, each window will be of size window_size, padded with -1
    # for token of index i, w_i the window is: ([w_i-2,w_i-1 i, w_i+1,w_i+2],label of w_i)
    windows = []
    windows_dict = {}
    tokens_idx_all = []
    labels_idx_all = []
    og_tokens = []
    for sentence in sentences:
        tokens, labels = sentence
        og_tokens.extend(tokens)
        # map tokens to their index in the vocabulary
        tokens_idx = [
            word_to_idx[word] if word in word_to_idx else word_to_idx["UUUNKKK"]
            for word in tokens
        ]
        tokens_idx_all.extend(tokens_idx)
        if type != "test":
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
            # label = labels_idx[i]
            label = ""
            windows.append((context, label))
            # windows_dict[i] = (context, label)
    # TODO maybe we don't need to calculate it and return it ?
    tokens_idx = torch.tensor(tokens_idx_all)
    if type != "test":
        labels_idx = torch.tensor(labels_idx_all)
    else:
        labels_idx = torch.tensor(
            [0] * len(tokens_idx_all)
        )  # fake labels for compatibility
    return tokens_idx, labels_idx, windows, vocab, labels_vocab, windows_dict, og_tokens


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
    (
        tokens_idx,
        labels_idx,
        windows,
        vocab,
        labels_vocab,
        windows_dict,
        og_tokens,
    ) = read_data(f"./{task}/train", task=task)
    # create an empty embedding matrix, each vector is size 50
    embedding_matrix = nn.Embedding(len(vocab), 50, _freeze=False)
    embedding_matrix.weight.requires_grad = True

    # initialize the embedding matrix to random values using xavier initialization which is a good initialization for NLP tasks
    nn.init.xavier_uniform_(embedding_matrix.weight)

    model = Tagger(task, vocab, labels_vocab, embedding_matrix)

    # Make a new tensor out of the windows, so the tokens are windows of size window_size in the dataset
    token_idx_new = torch.tensor([window for window, label in windows])

    dataset = TensorDataset(token_idx_new, labels_idx)

    # Load the dev data
    (
        tokens_idx_dev,
        labels_idx_dev,
        windows_dev,
        vocab,
        labels_vocab,
        windows_dict,
        og_tokens,
    ) = read_data(f"./{task}/dev", task=task, vocab=vocab, labels_vocab=labels_vocab)

    token_idx_dev_new = torch.tensor([window for window, label in windows_dev])
    dev_dataset = TensorDataset(token_idx_dev_new, labels_idx_dev)
    # Get the dev loss from the model training
    dev_loss, dev_accuracy, dev_accuracy_no_o = train_model(
        model,
        input_data=dataset,
        dev_data=dev_dataset,
        epochs=1,
        lr=0.005,
        task=task,
    )

    plot_results(dev_loss, dev_accuracy, dev_accuracy_no_o, task)

    print("Test")
    (
        tokens_idx_test,
        labels_idx_test,
        windows_test,
        vocab_test,
        labels_vocab_test,
        windows_dict_test,
        og_tokens,
    ) = read_data(f"./{task}/test", type="test", vocab=vocab, labels_vocab=labels_vocab)

    token_idx_test_new = torch.tensor([window for window, label in windows_test])
    test_dataset = TensorDataset(token_idx_test_new, labels_idx_test)
    print()
    print("Test")

    # test_data
    # dev_loss, dev_acc, dev_acc_clean = test_model(
    #     model, test_dataset, windows, task="ner"
    # )
    run_inference(model, test_dataset, task, og_tokens)

    # print(f"Test Loss: {dev_loss}, Test Acc: {dev_acc} Acc No O:{dev_acc_clean}")
    dev_loss, dev_acc, dev_acc_clean = test_model(model, dev_dataset, windows, task="ner")

    print(
        f"Test Loss: {dev_loss}, Test Acc: {dev_acc} Acc No O:{dev_acc_clean}"
    )
    tokens_idx_test, labels_idx_test, windows_test, vocab_test, labels_vocab_test, windows_dict_test = read_data(
        "./ner/test", type="test"
    )

    # tokens_idx, labels_idx, windows, vocab, labels_vocab = read_data("./ner/test")


if __name__ == "__main__":
    tasks = ["ner", "pos"]
    for task in tasks:
        main(task)
