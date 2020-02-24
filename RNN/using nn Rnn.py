import torch
import torch.nn as nn

input_size = 5
hidden_size = 8

inputs = torch.Tensor(1,10,5)

cell = nn.RNN(input_size, hidden_size, batch_first=True)

outputs, _status = cell(inputs)

print(outputs.shape)
print(_status.shape) 