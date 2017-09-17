# The Following Program is Designed for the Istanbul Stock Market Forecasting.
# Project Designed and Maintained by Mrinal Wahal.

import numpy as np

def sigmoid(x, derivative = False):
    if derivative == True: return x(x-1)
    return (1/(1 + np.exp(-x)))

def think(input_layer, syn0, syn1):
    l1 = sigmoid(np.dot(input_layer,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    return l2

def train(input_layer, output_layer, syn0, syn1):
    for j in xrange(100000):
        l1 = sigmoid(np.dot(input_layer,syn0))
        l2 = sigmoid(np.dot(l1,syn1))
        l2_delta = (output_layer - l2)*(l2*(1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += np.dot(l1,l2_delta)
        syn0 += input_layer.T.dot(l1_delta)

def main():
    
    X = np.array([[0.035753708, 0.038376187, -0.004679315, 0.002193419, 0.003894376, 0,\
        0.031190229, 0.012698039, 0.028524462]])
    y = np.array([[0.025425873,0.031812743,0.007786738,0.008455341,0.012865611,\
               0.004162452,0.01891958,0.011340652,0.008772644]])

    input_nodes = 9
    global syn0
    syn0 = np.zeros((1, input_nodes)).T
    global syn1
    syn1 = np.zeros((input_nodes,1)).T

    print "-"*60
    print "Before Training."
    print think(X, syn0, syn1)
    print "After Training."
    train(X,y, syn0, syn1)
    print think(X, syn0, syn1)
    print "-"*60

if __name__ == "__main__": main ()

"""
print "New Inputs."
x1 = np.array([ [1,100], [100,100], [1,1], [100,1]])
print think(x1, syn0, syn1)
"""
