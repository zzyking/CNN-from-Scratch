from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch
import os

def load_dataset(load_path):
  return torch.load(load_path)

def get_data_loaders(data_dir, batch_size, data_augmentation=False):

  train_dataset_path = os.path.join(data_dir, 'processed_train_dataset.pt')
  augmented_dataset_path = os.path.join(data_dir, 'processed_augmented_dataset.pt')
  test_dataset_path = os.path.join(data_dir, 'processed_test_dataset.pt')

  train_dataset = load_dataset(train_dataset_path)
  augmented_dataset = load_dataset(augmented_dataset_path)
  test_dataset = load_dataset(test_dataset_path)


  if data_augmentation and augmented_dataset:
    train_dataset = ConcatDataset([train_dataset, augmented_dataset])

  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  return train_loader, test_loader
