from torchvision import datasets, transforms
import torch
import os
from config import Config

def get_data_transforms(data_augmentation=False):
  if data_augmentation:
    transform_train = transforms.Compose([
      transforms.ToTensor(),
      transforms.RandomHorizontalFlip(),
      transforms.RandomRotation(15),
      transforms.RandomAdjustSharpness(sharpness_factor=2),
      transforms.RandomAutocontrast(),
      transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
      transforms.RandomGrayscale(p=0.1),
      transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
      transforms.RandomErasing(p=0.2, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
  else:
    transform_train = transforms.Compose([
      transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

  transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
  ])

  return transform_train, transform_test

def save_dataset(dataset, save_path):
  torch.save(dataset, save_path)

if __name__ == '__main__':

  config = Config()

  transform_train, transform_test = get_data_transforms(False)
  transform_augmented, _ = get_data_transforms(True)

  train_dataset_path = os.path.join(config.data_dir, 'processed_train_dataset.pt')
  augmented_dataset_path = os.path.join(config.data_dir, 'processed_augmented_dataset.pt')
  test_dataset_path = os.path.join(config.data_dir, 'processed_test_dataset.pt')

  train_dataset = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=transform_train)
  test_dataset = datasets.CIFAR10(root=config.data_dir, train=False, download=True, transform=transform_test)
  augmented_dataset = datasets.CIFAR10(root=config.data_dir, train=True, download=True, transform=transform_augmented)

  save_dataset(train_dataset, train_dataset_path)
  save_dataset(augmented_dataset, augmented_dataset_path)
  save_dataset(test_dataset, test_dataset_path)

  print(f"Processed datasets saved to {config.data_dir}")
