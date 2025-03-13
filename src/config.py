# config.py
import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
NUM_EPOCHS = 100
BATCH_SIZE = 10
LEARNING_RATE = 0.01

# Linear Regression Dataset parameters
NUM_SAMPLES = 100

# Model parameters for linear regression
INPUT_DIM = 1
OUTPUT_DIM = 1

