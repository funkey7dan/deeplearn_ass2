# Import the necessary libraries
import torch
import torch.nn as nn
from tagger4 import Tagger, WordCNN
import matplotlib.pyplot as plt
import math

# Set the task as named entity recognition (ner)
task = "ner"

# Load the pre-trained model
model = torch.load(f"model_{task}.pt")

# Find the convolutional layer in the model
conv_layer = None
for module in model.modules():
    if isinstance(module, nn.Conv1d):
        conv_layer = module
        break

# Get the character vocabulary
char_to_idx = model.char_to_idx
idx_to_char = {i: char for char, i in char_to_idx.items()}
char_vocab_size = len(char_to_idx)

# Set the batch size, sequence length, and input channels
batch_size = 32
sequence_length = 100
input_channels = 30

# Create an input sequence of 54 characters
sequence = "This is a sequence of characters"
word = sequence[:54].lower()

# Get the maximum word length allowed by the model
max_word_len = model.word_cnn.max_word_len

# Add padding to the input sequence to make it the same length as the maximum word length
padding_front = math.floor((max_word_len - len(word)) / 2)
padding_back = max_word_len - len(word) - padding_front
word = [model.char_to_idx[char] for char in word]
word = [model.char_embedding(torch.tensor(char)) for char in word]
word = torch.stack(word)
word_matrix = nn.functional.pad(word, (0, 0, padding_front, padding_back))
embedding = word_matrix.permute(1, 0)

# Pass the input sequence through the convolutional layer
output = conv_layer(embedding)


"""
The output of the activation map is then plotted to visualize how the convolutional layer is processing the input sequence.
The x-axis of the plot shows the characters in the input sequence, while the y-axis shows the output of the i'th filter.
"""

# Plot the output of the activation map
plt.imshow(output.squeeze().detach().numpy())

# Set the x-axis ticks and labels
tick_positions = range(padding_front, padding_front + len(sequence[:54]))
tick_labels = list(sequence[:54])
plt.xticks(tick_positions, tick_labels)

# Show the plot
plt.show()
