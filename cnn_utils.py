import numpy as np
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)

# data loader function, return four datasets: train_data_x, train_data_y, test_data_x, test_data_y
def load_dataset():
    with open('datasets/train_data_x.pkl','rb') as f:
        train_data_x = pkl.load(f, encoding='latin1')

    with open('datasets/train_data_y.pkl','rb') as f:
        train_data_y = pkl.load(f)

    with open('datasets/test_data_x.pkl','rb') as f:
        test_data_x = pkl.load(f, encoding='latin1')

    with open('datasets/test_data_y.pkl','rb') as f:
        test_data_y = pkl.load(f)

    return train_data_x, train_data_y, test_data_x, test_data_y

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # First convolution layer with output channel = 6, kernel size = 5x5
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        # Max pooling layer following each convolution layer with filter size = 2x2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Second convolution layer with output channel = 12, kernel size = 5x5
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, stride=1)

        # Fully connected layers
        self.fc1 = nn.Linear(in_features=12 * 13 * 13, out_features=120)  # First fully connected layer
        self.fc2 = nn.Linear(in_features=120, out_features=64)  # Second fully connected layer
        self.fc3 = nn.Linear(in_features=64, out_features=2)   # Third fully connected layer

    def forward(self, x):
        # Apply convolution, activation (ReLU), and pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten the output for the fully connected layers
        x = torch.flatten(x, 1) # flatten all dimensions except batch

        # Apply fully connected layers with ReLU activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Output layer with sigmoid activation
        x = torch.sigmoid(self.fc3(x))
        return x