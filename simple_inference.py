import torch
import torch.nn as nn
import re
from torch.nn.utils.rnn import pad_sequence
import numpy as np

# Load state_dict from linear predictor model
state_dict = torch.load('linear_predictor_L14_MSE.pth', map_location=torch.device('cpu'))

# Extract vocabulary size, embedding size, and number of classes from state_dict
vocab_size = state_dict['layers.0.weight'].size(0)
embedding_size = state_dict['layers.0.weight'].size(1)
num_classes = state_dict['layers.2.weight'].size(1)

# Define input, hidden, and output sizes for the model
input_size = vocab_size
hidden_size = embedding_size
output_size = num_classes

# Define the model
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Create an instance of the model
model = MyModel(input_size, hidden_size, output_size)

# Load the state_dict into the model
state_dict_new = model.state_dict()
for k, v in state_dict.items():
    if k in state_dict_new:
        state_dict_new[k] = v

model.load_state_dict(state_dict_new)
model.eval()

# Extract the weights and bias of the first and second linear layers
weights_1 = model.state_dict()['fc1.weight']
weights_2 = model.state_dict()['fc2.weight']
bias_1 = model.state_dict()['fc1.bias']
bias_2 = model.state_dict()['fc2.bias']


# Tokenize input text
def tokenize(text):
    text = re.sub(r'[^a-zA-Z]+', ' ', text)
    text = text.lower()
    tokens = text.split()

    return tokens

# Tokenize test data
test_data = ["Im a sentence"] 
test_data_tokens = [tokenize(text) for text in test_data]

# Create a vocabulary and index it
vocab = set([token for tokens in test_data_tokens for token in tokens])
vocab_index = {token: i for i, token in enumerate(vocab)}

# Replace tokens with their indexed values
test_data_tokens_indexed = [[vocab_index[token] for token in tokens] for tokens in test_data_tokens]

# Determine the maximum length of the tokenized inputs
max_len = max(len(tokens) for tokens in test_data_tokens_indexed)
test_data_tokens_indexed = [torch.tensor(tokens) for tokens in test_data_tokens_indexed]

for i, tokens in enumerate(test_data_tokens_indexed):
  num_padding = max_len - len(tokens)
  padding = torch.zeros(num_padding, dtype=torch.long)
  test_data_tokens_indexed[i] = torch.cat((tokens, padding))

test_data_tokens_indexed_tensor = torch.stack(test_data_tokens_indexed)
weights_1_shape = model.state_dict()['fc1.weight'].shape

X = torch.zeros(len(test_data_tokens_indexed_tensor), weights_1_shape[1]).type_as(weights_1)

predictions = model(X)
predictions = torch.abs(predictions)
predictions = torch.sqrt(predictions)

output_mean = torch.mean(predictions)
print("Output von der Prediction:", output_mean)
