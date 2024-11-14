import argparse
from models.simple_cnn import SimpleCNN
from models.deep_cnn import DeepCNN
from utils.data_loader import get_data_loaders
from utils.train import train
from utils.test import test
from utils.visualize import plot_train_loss_and_test_acc
from config import Config
import torch.nn as nn
import torch.optim as optim

def main():
  parser = argparse.ArgumentParser(description='Train a CNN on CIFAR-10')
  parser.add_argument('--model', type=str, default='simple', choices=['simple', 'deep'], help='Model type')
  parser.add_argument('--normalization', type=str, default=None, choices=[None, 'batch', 'layer'], help='Normalization method')
  parser.add_argument('--data_augmentation', action='store_true', help='Use data augmentation')
  parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay (L2 regularization)')
  parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
  args = parser.parse_args()

  config = Config()

  train_loader, test_loader = get_data_loaders(config.data_dir, config.batch_size, data_augmentation=args.data_augmentation)

  if args.model == 'simple':
    model = SimpleCNN(normalization=args.normalization, dropout_rate=args.dropout_rate).to(config.device)
  elif args.model == 'deep':
    model = DeepCNN(normalization=args.normalization, dropout_rate=args.dropout_rate).to(config.device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=args.weight_decay)

  train_losses = []
  test_accs = []

  for epoch in range(1, config.num_epochs + 1):
    train(model, config.device, train_loader, optimizer, criterion, epoch, train_losses)
    test(model, config.device, test_loader, criterion, test_accs)

  plot_train_loss_and_test_acc(train_losses, test_accs, args, save_dir=config.fig_dir)

if __name__ == '__main__':
    main()
