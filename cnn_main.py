import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import cnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)

def model_train(train_data_x, train_data_y, test_data_x, test_data_y):
    net = U.Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    train_data_x = torch.tensor(train_data_x, dtype=torch.float32).permute(0,3,1,2)
    train_data_y = torch.tensor(train_data_y, dtype=torch.long)
    test_data_x = torch.tensor(test_data_x, dtype=torch.float32)
    test_data_y = torch.tensor(test_data_y, dtype=torch.long)
    
    test_data_y = test_data_y.view(-1)
    train_data_y = train_data_y.view(-1)
    train_dataset = TensorDataset(train_data_x, train_data_y)
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)

    for epoch in range(10):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
  
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/10], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
    print('Training complete')
    model_test(test_data_x, test_data_y, net, 10)
    
# model test: can be called directly in model_train 
def model_test(test_data_x, test_data_y, net, epoch_num):
    # Convert numpy arrays to torch tensors
    test_data_x = torch.tensor(test_data_x, dtype=torch.float32).permute(0,3,1,2)
    test_data_y = torch.tensor(test_data_y, dtype=torch.long)

    # Create DataLoader for the test data
    test_dataset = TensorDataset(test_data_x, test_data_y)
    test_loader = DataLoader(test_dataset, batch_size=5)

    # Ensure the model is in evaluation mode
    net.eval()

    correct = 0
    total = 0

    # No need to track gradients for testing
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the network on the test images after epoch {epoch_num}: {accuracy:.2f}%')

    # Optionally return the accuracy
    return accuracy

if __name__ == '__main__':
	# load datasets
	train_data_x, train_data_y, test_data_x, test_data_y = U.load_dataset()

	# rescale data 
	train_data_x = train_data_x / 255.0
	test_data_x = test_data_x / 255.0

	# model train (model test function can be called directly in model_train)
	model_train(train_data_x, train_data_y, test_data_x, test_data_y)










