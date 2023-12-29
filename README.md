**Convolutional Neural Network **
Build a convolutional neural network model for image
classification (binary classification) using Pytorch (https://pytorch.org/). 
Finished the code in cnn_main.py and cnn_utils.py,

Four datasets (pkl format) in the datasets folder:
train_data_x.pkl: training data
train_data_y.pkl: label of training data
test_data_x.pkl: test data
test_data_y.pkl: label of test data

Finished Net class for constructing CNN model in cnn_utils.py.
CNN model detail: 5 layers in total. The first 2 layers are convolution layers with
kernel size = 5*5, stride = 1, and output channel numbers = 6, 12. 

The filter size of
max pooling layer after each convolution layer is 2*2. The last 3 layers are fully
connected layers with output hidden numbers = 120, 64, and 2, respectively. Use
relu activation in each hidden layer and sigmoid activation in the last layer. Use
default parameter initialization in Pytorch.

Finished model_train function for CNN model training (using train data)
with Adam optimization in cnn_main.py. Set batch size = 5.

Finished model_test function for model testing (using test data) in cnn_main.py.
