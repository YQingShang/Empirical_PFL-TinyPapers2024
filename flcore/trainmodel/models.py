import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor

batch_size = 10

# ====================================================================================================================

class Mclr_Logistic(nn.Module):
    def __init__(self, input_dim=1*31, num_classes=2):
        super(Mclr_Logistic, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output

# ====================================================================================================================

class BinaryLogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(BinaryLogisticRegression, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        output = self.sigmoid(x)
        return output

# ====================================================================================================================

