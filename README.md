# Neural-network

This is a code for neural network from scratch. It uses only the numpy.
You can use sigmoid, tahn activation functions and SSE, CE loss functions.

You can set the number of hidden layers and the number of nodes in each hidden layers.

This code can only use one activation function for each training.

Command:
python NN.py -A train data.txt -y train target.txt -ln layer number -un n1,n2,n3,..,nln -a activation -ls loss -out output.txt -lr learning rate -nepochs max epochs -bs batch size -tol tol

It creates a txt file that contains avg of loss function values.
avg value is created in every batch.
