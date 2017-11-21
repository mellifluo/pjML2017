import numpy as np
import csv

#load dataset
X = np.loadtxt('./monk/monks-1.train', dtype='string', delimiter=' ')
#output
y = X[:,1].astype(int)
#input
X = X[:,2:-1].astype(int)
#Sigmoid Function
def sigmoid (x):
    return 1/(1 + np.exp(-x))

#Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

#Variable initialization
epoch=1000 #Setting training iterations
lr=0.1 #Setting learning rate
inputlayer_neurons = X.shape[1] #number of features in data set
hiddenlayer_neurons = 3 #number of hidden layers neurons
output_neurons = 1 #number of neurons at output layer

#weight and bias initialization
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,))
bout=np.random.uniform(size=(1,output_neurons))

for i in range(epoch):

    #Forward Propogation
    hidden_layer_input1=np.dot(X,wh) #muliply weights and input matrix obtaining the first hidden layer input
    hidden_layer_input=hidden_layer_input1 + bh #biased
    hiddenlayer_activations = sigmoid(hidden_layer_input)  #activation function on input
    output_layer_input1=np.dot(hiddenlayer_activations,wout) #same shit
    output_layer_input= output_layer_input1+ bout
    output = sigmoid(output_layer_input)[:,-1]
    #Backpropagation
    E = (y-output)**2/2
    slope_output_layer = derivatives_sigmoid(output) #derivata della funzione di attivazione sull'output
    slope_hidden_layer = derivatives_sigmoid(hiddenlayer_activations) #derivata dell'hidden layer
    d_output = (E * slope_output_layer) #delta output layer
    error_at_hidden_layer = d_output.reshape(d_output.shape + (1,)).dot(wout[np.newaxis])
    d_hiddenlayer = error_at_hidden_layer * slope_hidden_layer #delta hidden layer
    wout += hiddenlayer_activations.T.dot(d_output) *lr #aggiorno pesi output
    bout += np.sum(d_output, axis=0,keepdims=True) *lr #aggiorno bias output
    wh += X.T.dot(d_hiddenlayer) *lr #aggiorno pesi hidden
    bh += np.sum(d_hiddenlayer, axis=0,keepdims=True) *lr #aggiorno bias hidden

print output[0]
