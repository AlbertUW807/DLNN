# Libraries
import numpy as np
from testCases import *
from gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, gradients_to_vector

# 1-Dimensional Gradient Check
# ======================================================================================================
def forward_propagation(x, theta):
    J = theta * x
    return J

def backward_propagation(x, theta):
    dtheta = x
    return dtheta

def gradient_check(x, theta, epsilon = 1e-7):
    thetaplus = theta + epsilon                              
    thetaminus = theta - epsilon                             
    J_plus = forward_propagation(x, thetaplus)                                 
    J_minus = forward_propagation(x, thetaminus)                                
    gradapprox = (J_plus - J_minus) / ( 2. * epsilon)                      
    grad = backward_propagation(x, theta)
    numerator = np.linalg.norm(grad - gradapprox)                              
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)                           
    difference = numerator / denominator                        
  
    if difference < 1e-7: # condition to be satisfied
        print ("The gradient is correct!")
    else:
        print ("The gradient is wrong!")
    
    return difference
# ======================================================================================================


# N-Dimensional Gradient Check
# ======================================================================================================
def forward_propagation_n(X, Y, parameters):
    m = X.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1 - A3), 1 - Y)
    cost = 1./m * np.sum(logprobs)
    
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    
    return cost, cache
# ======================================================================================================