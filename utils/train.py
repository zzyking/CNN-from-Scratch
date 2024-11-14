import torch
import torch.nn as nn
import torch.optim as optim

def train(model, device, train_loader, optimizer, criterion, epoch, train_losses):
  model.train()
  running_loss = 0.0
  for batch_idx, (data, target) in enumerate(train_loader):
    data, target = data.to(device), target.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    running_loss += loss.item()
    if batch_idx % 100 == 0:
      print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
  avg_loss = running_loss / len(train_loader)
  train_losses.append(avg_loss)
  print(f'Train Epoch: {epoch} Average Loss: {avg_loss:.6f}')