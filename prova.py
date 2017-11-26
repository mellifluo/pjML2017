import numpy as np
import csv
import matplotlib.pyplot as plt

#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#load dataset
X = np.loadtxt('./monk/monks-1.train', dtype='string', delimiter=' ')
#output
y = X[:,1].astype(int)
y = y[..., np.newaxis]
#input
X = X[:,2:-1].astype(int)
#Variable initialization
epoch=10000 #Setting training iterations
lr=0.2 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(size=(1,output_neurons))
for i in range(epoch):

    #Forward Propogation
    # livello intermedio
    hidden_layer_input1 = np.dot(X,wh) #muliply weights and input matrix obtaining the first hidden layer input
    hidden_layer_input = hidden_layer_input1 + bh #biased
    hiddenlayer_activations = sigmoid(hidden_layer_input)  #activation function on input
    # livello output
    output_layer_input1 = np.dot(hiddenlayer_activations,wout) #same shit
    output_layer_input = output_layer_input1 + bout
    output = sigmoid(output_layer_input)
    #Backpropagation
    # livello output
    # E = np.divide(np.power(np.subtract(y,output),2),2)
    E = np.subtract(y,output)
    slope_output_layer = derivatives_sigmoid(output) #derivata della funzione di attivazione sull'output
    d_output = np.multiply(slope_output_layer,E) #delta output layer
    wout += np.dot(hiddenlayer_activations.T,(d_output)) * lr #aggiorno pesi output layer
    # bout += np.sum(d_output, axis=0,keepdims=True) *lr #aggiorno bias output layer
    # livello intermedio
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations) #derivata dell'hidden layer
    error_at_hidden_layer = np.dot(d_output,wout.T)
    d_hiddenlayer = np.multiply(error_at_hidden_layer,slope_hidden_layer) #delta hidden layer
    wh += np.dot(X.T,d_hiddenlayer) * lr #aggiorno pesi hidden
    # bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr #aggiorno bias hidden

plt.plot(y)
plt.plot(output)
