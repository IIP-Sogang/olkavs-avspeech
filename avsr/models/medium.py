import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_1 = nn.Linear(input_dim, output_dim*4)
        self.linear_2 = nn.Linear(output_dim*4, output_dim)
        self.batchnorm = nn.BatchNorm1d(output_dim * 4)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        batch_seq_size = x.shape[:2]
        output = self.linear_1(x)
        output = output.permute((0,2,1))
        output = self.batchnorm(output)
        output = output.permute((0,2,1))
        output = self.relu(output)
        output = self.linear_2(output)
        output = output.view(*batch_seq_size, -1)
        return output