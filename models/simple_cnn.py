import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
  def __init__(self, normalization=None, dropout_rate=0.5):
    super(SimpleCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 8 * 8, 128)
    self.fc2 = nn.Linear(128, 10)
    self.dropout = nn.Dropout(dropout_rate)

    if normalization == 'batch':
      self.norm1 = nn.BatchNorm2d(32)
      self.norm2 = nn.BatchNorm2d(64)
    elif normalization == 'layer':
      self.norm1 = nn.LayerNorm([32, 32, 32])
      self.norm2 = nn.LayerNorm([64, 16, 16])
    else:
      self.norm1 = nn.Identity()
      self.norm2 = nn.Identity()

  def forward(self, x):
    x = self.pool(F.relu(self.norm1(self.conv1(x))))
    x = self.pool(F.relu(self.norm2(self.conv2(x))))
    x = x.view(-1, 64 * 8 * 8)
    x = self.dropout(F.relu(self.fc1(x)))
    x = self.fc2(x)
    return x
