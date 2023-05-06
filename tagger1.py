import torch
import torch.nn as nn
import numpy as np
import os


class Tagger:
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

    def __init__(self):
        self.model = nn.Sequential()


def read_data(fname):
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
    # Convert the tokens and labels into NumPy arrays
    tokens = np.array(tokens[1:])
    labels = np.array(labels[1:])
    print()
    return tokens, labels


def main():
    tokens, labels = read_data("./ner/train")


if __name__ == "__main__":
    main()
