import torch

class Config:
  def __init__(self):
    self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    self.batch_size = 64
    self.num_epochs = 10
    self.learning_rate = 0.001
    self.data_dir = './data'
    self.model_dir = './models'
