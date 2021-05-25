import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from os import listdir
from os.path import isfile, join


def get_array(folder):
    image_list = [f for f in listdir(folder) if isfile(join(folder, f))]
    m = np.array([])
    for i_name in image_list:
        img = Image.open(folder+i_name)
        p = np.array(img.resize((256,256), Image.ANTIALIAS))
        if len(p.shape) == 3:
            p = p.reshape(p.shape[0]*p.shape[1]*p.shape[2],1)
            if len(m)==0:
                m = p
            else:
                m = np.concatenate((m,p),axis=1)
    return m


def initialization():

    # Training Data Set
    train_WBC = get_array('segmentation_WBC-master/segmentation_WBC-master/Dataset 1/training/')
    train_RBC = get_array('segmentation_WBC-master/RBCs/training/')
    train_WBC_result = np.ones((1,train_WBC.shape[1]))
    train_RBC_result = np.zeros((1,train_RBC.shape[1]))


    # Concatenation
    train_input = np.concatenate((train_WBC,train_RBC),axis=1)
    train_output = np.concatenate((train_WBC_result,train_RBC_result),axis=1)


    # Testing Data Set
    test_WBC = get_array('segmentation_WBC-master/segmentation_WBC-master/Dataset 1/testing/')
    test_RBC = get_array('segmentation_WBC-master/RBCs/testing/')
    test_WBC_result = np.ones((1,test_WBC.shape[1]))
    test_RBC_result = np.zeros((1,test_RBC.shape[1]))

    # Concatenation
    test_input = np.concatenate((test_WBC,test_RBC),axis=1)
    test_output = np.concatenate((test_WBC_result,test_RBC_result),axis=1)

    train_input = train_input / 225.
    test_input = test_input / 225.

    return train_input, train_output, test_input, test_output
    

# Sigmoid is the activtion function for the hidden and the output layer
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def layer_sizes(X, Y):
    n_i = X.shape[0]
    n_o = Y.shape[0]
    return (n_i,n_o)


def initialize_parameters(n_i, n_h, n_o):
    # W are the weight matrices
    # b are the biases
    # W1 is the weight matrix of layer 1
    # Xavier’s Initialization
    W1 = np.random.randn(n_h,n_i)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_o, n_h)*0.01
    b2 = np.zeros((n_o,1))
    parameters = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Sigmoid takes any real value as input and outputs values in the range -1 to 1 
    Z1 = np.dot(W1,X)+b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache


def compute_cost(A2, Y, parameters):
    # Gradient Descent
    m = Y.shape[1]
    # Loss
    logprobs = -np.sum(Y*np.log(A2)+(1-Y)*np.log(1-A2))
    cost = (1/m)*logprobs
    cost = np.squeeze(cost)
    
    return cost


# Used in creating the model
def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Compute the derivative 
    # Gradient Descent
    dZ2 = A2-Y
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2,axis=1,keepdims = True)
    dZ1 = np.dot(W2.T,dZ2)*(1-np.power(A1,2))
    dW1 = (1/m)*np.dot(dZ1,X.T)
    db1 = (1/m)*np.sum(dZ1,axis=1,keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads


def update_parameters(parameters, grads, learning_rate = 0.0005):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Gradients
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # new weight = old weight - learning rate * dl
    W1 = W1-learning_rate*dW1
    b1 = b1-learning_rate*db1
    W2 = W2-learning_rate*dW2
    b2 = b2-learning_rate*db2
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters


def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        
        if print_cost and i % 100 == 0:
            print ("Loss after iteration %i: %f" %(i, cost))

    return parameters


def predict(parameters, X):
    
    # Forward propagation outputs the activation function
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    
    return predictions


def shallow_neural_network():

    train_input, train_output, test_input, test_output = initialization()
    
    # Build a model with a n_h-dimensional hidden layer
    parameters = nn_model(train_input, train_output, n_h = 4, num_iterations = 200, print_cost=True)

    # Print accuracy
    predictions = predict(parameters, train_input)
    accuracy_train = float((np.dot(train_output,predictions.T) + np.dot(1-train_output,1-predictions.T))/float(train_output.size)*100)
    predictions = predict(parameters, test_input)
    accuracy_test = float((np.dot(test_output,predictions.T) + np.dot(1-test_output,1-predictions.T))/float(test_output.size)*100) 
    
    return accuracy_train, accuracy_test

