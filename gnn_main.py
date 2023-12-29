import numpy as np
# import matplotlib.pyplot as plt
# import scipy
import gnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(0)

# model train
def model_train(model, adj, features, labels, idx_train, idx_test, epochs=200, learning_rate=0.01):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss = F.cross_entropy(output[idx_train], labels[idx_train])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')
    model_test(model, adj, features, labels, idx_test)
	
def model_test(model, adj, features, labels, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss = F.cross_entropy(output[idx_test], labels[idx_test])
        pred = output[idx_test].max(1)[1]
        correct = pred.eq(labels[idx_test]).sum().item()
        acc = correct / idx_test.size(0)
    
    print(f'Test set results, Loss: {loss.item()}, Accuracy: {acc}')
    return loss, acc

if __name__ == '__main__':
	# load datasets
    adj, features, labels, idx_train, idx_test = U.load_data()
    model = U.GCN(features.shape[1], 32, 7, 0.5)
	# model train (model test function can be called directly in model_train
    model_train(model, adj, features, labels, idx_train ,idx_test)