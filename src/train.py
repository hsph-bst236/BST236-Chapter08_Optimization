# train.py
import torch
import torch.nn as nn
import datetime
import subprocess
import os
from utils import *

class Trainer:
    """
    A class to handle the training process of a PyTorch model.
    """
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device,
        num_epochs,
        use_logger=False,
        save_frequency=10,
        hyperparams=None,
        git_commit=False
    ):
        """
        Initialize the Trainer with model and training parameters.
        
        Args:
            model: PyTorch model to train
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            loss_fn: Loss function
            optimizer: Optimizer for parameter updates
            device: Device to use for computation
            num_epochs: Number of training epochs
            use_logger: Whether to use logging functionality
            save_frequency: How often to save model checkpoints (epochs)
            hyperparams: Dictionary of hyperparameters to log
            git_commit: Whether to automatically create a git commit after training
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.num_epochs = num_epochs
        self.use_logger = use_logger
        self.save_frequency = save_frequency
        self.hyperparams = hyperparams
        self.git_commit = git_commit
        
        # Extract save directories from hyperparameters if provided
        self.checkpoints_dir = hyperparams.get('CHECKPOINTS_DIR', 'checkpoints') if hyperparams else 'checkpoints'
        self.results_dir = hyperparams.get('RESULTS_DIR', 'results') if hyperparams else 'results'
        self.logs_dir = hyperparams.get('LOGS_DIR', 'logs') if hyperparams else 'logs'
        
        # Logger setup
        self.logger = None
        if self.use_logger:
            self.logger = Logger(log_dir=self.logs_dir)
            self.logger.start_capture()
            # Log timestamp at the beginning
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"Training started at: {timestamp}\n")
            # Log hyperparameters if provided
            if self.hyperparams:
                self.logger.log_hyperparameters(self.hyperparams)
        
        # Variables for tracking training progress
        self.train_losses = []
        self.val_losses = []
        
        # Store start time for commit message
        self.start_time = datetime.datetime.now()
    
    def _train_epoch(self):
        """Train the model for one epoch."""
        self.model.train()  # Set the model to training mode
        running_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # Move data to the specified device
            data = data.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            predictions = self.model(data)
            loss = self.loss_fn(predictions, targets)
            running_loss += loss.item()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx}/{len(self.train_loader)}: Loss = {loss.item():.4f}")
        
        avg_loss = running_loss / len(self.train_loader)
        return avg_loss
    
    def _save_checkpoint(self, epoch):
        """Save model checkpoint."""
        checkpoint = {
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()
        }
        save_checkpoint(checkpoint, filename=f"checkpoint_{epoch}.pth.tar", directory=self.checkpoints_dir)
        print(f"Saved checkpoint {epoch} with validation loss: {self.val_losses[-1]:.4f}")
    
    def _make_git_commit(self, final_train_loss, final_val_loss):
        """Create a git commit with training results."""
        try:
            # Check if this is a git repository
            subprocess.check_output(["git", "rev-parse", "--is-inside-work-tree"], stderr=subprocess.DEVNULL)
            
            # Format the training duration
            end_time = datetime.datetime.now()
            duration = end_time - self.start_time
            hours, remainder = divmod(duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            duration_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
            
            # Create commit message
            commit_time = end_time.strftime('%Y-%m-%d %H:%M:%S')
            commit_message = (
                f"Training completed at {commit_time}\n\n"
                f"Duration: {duration_str}\n"
                f"Final Training Loss: {final_train_loss:.6f}\n"
                f"Final Validation Loss: {final_val_loss:.6f}\n"
                f"Epochs: {self.num_epochs}\n"
                f"Learning Rate: {self.hyperparams.get('LEARNING_RATE', 'N/A') if self.hyperparams else 'N/A'}"
            )
            
            # Check if there are modified or new files in the output directories
            subprocess.run(["git", "add", self.checkpoints_dir, self.results_dir, self.logs_dir])
            
            # Create the commit
            subprocess.run(["git", "commit", "-m", commit_message])
            
            print("\nCreated git commit with training results.")
            print(f"Commit message:\n{commit_message}")
            
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"\nFailed to create git commit: {str(e)}")
            print("This might not be a git repository or git might not be installed.")
    
    def train(self):
        """Run the full training process."""
        print(f"Training on device: {self.device}")
        
        # Training loop over epochs
        for epoch in range(self.num_epochs):
            print(f"\nEpoch [{epoch+1}/{self.num_epochs}]")
            
            # Train for one epoch
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            # Evaluate on validation set
            val_loss = evaluate_model(self.val_loader, self.model, self.loss_fn, device=self.device)
            self.val_losses.append(val_loss)
            
            # Save model if needed
            if epoch % self.save_frequency == 0:
                self._save_checkpoint(epoch)
        
        # Log final training metrics
        if self.use_logger and self.logger:
            self.logger.log_final_metrics(self.train_losses[-1], self.val_losses[-1])
        
        # Visualize final model predictions
        visualize_predictions(self.model, device=self.device, save_dir=self.results_dir)
        
        # Plot training and validation loss curves
        plot_loss_curves(self.train_losses, self.val_losses, save_dir=self.results_dir)
        
        print("Training completed!")
        
        # Print log file path if logging was used
        if self.use_logger and self.logger:
            log_path = self.logger.get_log_file_path()
            print(f"Log file saved to: {log_path}")
            # Stop logging
            self.logger.stop_capture()
        
        # Make git commit if enabled
        if self.git_commit:
            self._make_git_commit(self.train_losses[-1], self.val_losses[-1])
    
    def get_log_file_path(self):
        """Get the path to the log file if logging is enabled."""
        if self.use_logger and self.logger:
            return self.logger.get_log_file_path()
        return None
