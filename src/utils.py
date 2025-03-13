# utils.py
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from dataset import LinearRegressionDataset
import datetime
import sys
import io

def save_checkpoint(state, filename="my_checkpoint.pth.tar", directory="checkpoints"):
    """Save model checkpoint."""
    print("=> Saving checkpoint")
    if not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state, os.path.join(directory, filename))

def load_checkpoint(checkpoint, model, optimizer=None):
    """Load model checkpoint."""
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer"])

def get_loaders(num_samples, batch_size):
    """
    Create data loaders for training and validation.
    
    Args:
        num_samples (int): Number of samples for each dataset.
        batch_size (int): Batch size.
        
    Returns:
        tuple: (train_loader, val_loader) - Data loaders for training and validation.
    """
    train_dataset = LinearRegressionDataset(num_samples=num_samples)
    val_dataset = LinearRegressionDataset(num_samples=num_samples // 4)  # Smaller validation set
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader

def evaluate_model(loader, model, loss_fn, device="cuda"):
    """
    Evaluate the model on the provided data loader.
    
    Args:
        loader (DataLoader): Data loader for evaluation.
        model (nn.Module): Model to evaluate.
        loss_fn (callable): Loss function.
        device (str): Device to use for computation.
        
    Returns:
        float: Average loss on the dataset.
    """
    model.eval()
    total_loss = 0.0
    num_batches = len(loader)
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            # Forward pass
            predictions = model(x)
            loss = loss_fn(predictions, y)
            
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    print(f"Evaluation - Avg Loss: {avg_loss:.4f}")
    return avg_loss

def visualize_predictions(model, device="cuda", num_points=100):
    """
    Visualize the model's predictions against the true relationship.
    
    Args:
        model (nn.Module): Trained model.
        device (str): Device to use for computation.
        num_points (int): Number of points to plot.
    """
    model.eval()
    
    # Generate test data across the range
    x = torch.linspace(-12, 12, num_points).view(-1, 1).to(device)
    
    # Get model predictions
    with torch.no_grad():
        predictions = model(x)
    
    # Convert to numpy for plotting
    x_np = x.cpu().numpy().flatten()
    predictions_np = predictions.cpu().numpy().flatten()
    
    # Generate true relationship (without noise)
    true_y = 3 * x_np + 2
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.scatter(x_np, predictions_np, color='blue', label='Model Predictions')
    plt.plot(x_np, true_y, color='red', label='True Relationship (y = 3x + 2)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression: Predictions vs. True Relationship')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/linear_regression_plot.png')
    plt.close()

def plot_loss_curves(train_losses, val_losses):
    """
    Plot the training and validation loss curves.
    
    Args:
        train_losses (list): Training losses per epoch.
        val_losses (list): Validation losses per epoch.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Add markers every 10 epochs for better readability
    marker_epochs = [i for i in epochs if i % 10 == 0 or i == 1 or i == len(epochs)]
    marker_train = [train_losses[i-1] for i in marker_epochs]
    marker_val = [val_losses[i-1] for i in marker_epochs]
    
    plt.plot(marker_epochs, marker_train, 'bo')
    plt.plot(marker_epochs, marker_val, 'ro')
    
    # Save the plot
    if not os.path.exists('results'):
        os.makedirs('results')
    plt.savefig('results/loss_curves.png')
    plt.close()
    
    print("Loss curves saved to 'results/loss_curves.png'")

class Logger:
    """
    A simple logger class that can redirect stdout to both console and a log file.
    """
    def __init__(self, log_dir='logs'):
        # Create logs directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        # Create a timestamped log file
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file_path = os.path.join(log_dir, f'training_log_{timestamp}.txt')
        self.log_file = open(self.log_file_path, 'w')
        
        # Store the original stdout
        self.original_stdout = sys.stdout
        
        # Initialize captured content
        self.captured_output = io.StringIO()
        
    def start_capture(self):
        """Start capturing stdout."""
        sys.stdout = self
        
    def stop_capture(self):
        """Stop capturing and restore original stdout."""
        sys.stdout = self.original_stdout
        self.log_file.close()
        
    def write(self, text):
        """Write to both stdout and log file."""
        self.original_stdout.write(text)
        self.log_file.write(text)
        self.captured_output.write(text)
        
    def flush(self):
        """Flush both stdout and log file."""
        self.original_stdout.flush()
        self.log_file.flush()
        
    def log_hyperparameters(self, config_dict):
        """Log hyperparameters to the log file."""
        self.write("\n" + "="*50 + "\n")
        self.write("HYPERPARAMETERS\n")
        self.write("="*50 + "\n")
        
        for key, value in config_dict.items():
            self.write(f"{key}: {value}\n")
            
        self.write("="*50 + "\n\n")
        
    def log_final_metrics(self, train_loss, val_loss):
        """Log final training metrics."""
        self.write("\n" + "="*50 + "\n")
        self.write("FINAL METRICS\n")
        self.write("="*50 + "\n")
        self.write(f"Final Training Loss: {train_loss:.6f}\n")
        self.write(f"Final Validation Loss: {val_loss:.6f}\n")
        self.write("="*50 + "\n\n")
        
    def get_log_file_path(self):
        """Return the path to the log file."""
        return self.log_file_path
