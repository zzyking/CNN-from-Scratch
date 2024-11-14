import torch.nn as nn
import torch.nn.functional as F

class DeepCNN(nn.Module):
  def __init__(self, normalization=None, dropout_rate=0.5):
    super(DeepCNN, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(128 * 4 * 4, 256)
    self.fc2 = nn.Linear(256, 128)
    self.fc3 = nn.Linear(128, 10)
    self.dropout = nn.Dropout(dropout_rate)

    if normalization == 'batch':
      self.norm1 = nn.BatchNorm2d(32)
      self.norm2 = nn.BatchNorm2d(64)
      self.norm3 = nn.BatchNorm2d(128)
    elif normalization == 'layer':
      self.norm1 = nn.LayerNorm([32, 32, 32])
      self.norm2 = nn.LayerNorm([64, 16, 16])
      self.norm3 = nn.LayerNorm([128, 8, 8])
    else:
      self.norm1 = nn.Identity()
      self.norm2 = nn.Identity()
      self.norm3 = nn.Identity()

  def forward(self, x):
    x = self.pool(F.relu(self.norm1(self.conv1(x))))
    x = self.pool(F.relu(self.norm2(self.conv2(x))))
    x = self.pool(F.relu(self.norm3(self.conv3(x))))
    x = x.view(-1, 128 * 4 * 4)
    x = self.dropout(F.relu(self.fc1(x)))
    x = self.dropout(F.relu(self.fc2(x)))
    x = self.fc3(x)
    return x
