# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import datetime
from model import LinearRegressionModel
from utils import save_checkpoint, get_loaders, evaluate_model, visualize_predictions, plot_loss_curves, Logger
from config import (
    DEVICE, NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, 
    NUM_SAMPLES, INPUT_DIM, OUTPUT_DIM
)

# Training function: iterate through the loader, perform forward and backward passes.
def train_fn(loader, model, optimizer, loss_fn, device):
    """
    Train the model for one epoch.
    
    Args:
        loader (DataLoader): Training data loader.
        model (nn.Module): Model to train.
        optimizer (Optimizer): Optimizer for parameter updates.
        loss_fn (callable): Loss function.
        device (str): Device to use for computation.
        
    Returns:
        float: Average training loss for the epoch.
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0
    
    for batch_idx, (data, targets) in enumerate(loader):
        # Move data to the specified device
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)
        running_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(loader)}: Loss = {loss.item():.4f}")
    
    avg_loss = running_loss / len(loader)
    return avg_loss

# Main function: setup model, loss function, optimizer, data loaders, then run training loop.
def main():
    # Initialize logger
    logger = Logger()
    logger.start_capture()
    
    # Log timestamp at the beginning
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"Training started at: {timestamp}\n")
    
    # Log hyperparameters
    hyperparams = {
        'DEVICE': DEVICE,
        'NUM_EPOCHS': NUM_EPOCHS,
        'BATCH_SIZE': BATCH_SIZE,
        'LEARNING_RATE': LEARNING_RATE,
        'NUM_SAMPLES': NUM_SAMPLES,
        'INPUT_DIM': INPUT_DIM,
        'OUTPUT_DIM': OUTPUT_DIM
    }
    logger.log_hyperparameters(hyperparams)
    
    print(f"Using device: {DEVICE}")
    
    # Create data loaders for training and validation
    train_loader, val_loader = get_loaders(
        num_samples=NUM_SAMPLES,
        batch_size=BATCH_SIZE
    )

    # Initialize the model
    model = LinearRegressionModel(INPUT_DIM, OUTPUT_DIM).to(DEVICE)
    # print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    
    # Variables for tracking training progress
    train_losses = []
    val_losses = []

    # Training loop over epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # Train for one epoch
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, DEVICE)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss = evaluate_model(val_loader, model, loss_fn, device=DEVICE)
        val_losses.append(val_loss)
        
        # Save model if it's the best so far
        if epoch % 10 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }
            save_checkpoint(checkpoint, filename=f"checkpoint_{epoch}.pth.tar")
            print(f"Saved checkpoint {epoch} with validation loss: {val_loss:.4f}")
    
    # Log final training metrics
    logger.log_final_metrics(train_losses[-1], val_losses[-1])
    
    # Visualize final model predictions
    visualize_predictions(model, device=DEVICE)
    
    # Plot training and validation loss curves
    plot_loss_curves(train_losses, val_losses)
    
    print("Training completed!")
    print(f"Log file saved to: {logger.get_log_file_path()}")
    
    # Stop logging
    logger.stop_capture()

if __name__ == "__main__":
    main()
