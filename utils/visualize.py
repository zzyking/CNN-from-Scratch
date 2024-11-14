import matplotlib.pyplot as plt
import os
from datetime import datetime

def plot_train_loss_and_test_acc(train_losses, test_accs, args, save_dir):

  exp_name = f"{args.model}_{args.normalization if args.normalization else 'no_norm'}"
  if args.data_augmentation:
      exp_name += "_augmented"
  if args.weight_decay > 0:
      exp_name += f"_wd{args.weight_decay}"
  exp_name += f"_drop{args.dropout_rate}"

  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  exp_name += f"_{timestamp}"

  # save_dir = os.path.join('./figs', args.model)
  os.makedirs(save_dir, exist_ok=True)

  plt.figure(figsize=(12, 8))

  plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train loss', color='blue')
  plt.xlabel('Epochs')
  plt.ylabel('Loss')
  plt.title('Training loss')
  plt.legend()

  plt.twinx()
  plt.plot()
  plt.plot(range(1, len(train_losses) + 1), test_accs, label='Test accuracy', color='red')
  plt.ylabel('Accuracy (%)')
  plt.legend(loc='upper right')

  save_path = os.path.join(save_dir, f"{exp_name}.png")
  plt.savefig(save_path)
  print(f"Plot saved to {save_path}")

  metrics_path = os.path.join(save_dir, f"{exp_name}_metrics.txt")
  with open(metrics_path, 'w') as f:
    f.write(f"Configuration:\n")
    f.write(f"Model: {args.model}\n")
    f.write(f"Normalization: {args.normalization}\n")
    f.write(f"Data Augmentation: {args.data_augmentation}\n")
    f.write(f"Weight Decay: {args.weight_decay}\n")
    f.write(f"Dropout Rate: {args.dropout_rate}\n\n")
    f.write(f"Final Test Accuracy: {test_accs[-1]:.2f}%\n")
    f.write(f"Best Test Accuracy: {max(test_accs):.2f}%\n")
    f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
