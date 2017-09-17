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

def normalize(value): return (float(value - min_val)/float(max_val - min_val))

def train(input_layer, output_layer, syn0, syn1):
    for iter in xrange(10000):
        l1 = sigmoid(np.dot(input_layer,syn0))
        l2 = sigmoid(np.dot(l1,syn1))
        l2_delta = (output_layer - l2)*(l2*(1-l2))
        l1_delta = l2_delta.dot(syn1.T) * (l1 * (1-l1))
        syn1 += np.dot(l1,l2_delta)
        syn0 += input_layer.T.dot(l1_delta)

def main():

    np.set_printoptions(suppress = True)
    x = np.array([[10101.05, 10128.60, 10065.75, 10114.65, 190000516, 11515.29]])
    X = ((x - x.min())/(x.max() - x.min()))
    y = np.array([[10136.30, 10137.85, 10054.20, 10081.50, 166463276, 9165.92]])
    
    input_nodes = 6
    global max_val
    max_val = x.max()
    global min_val
    min_val = x.min()

    #temp = []
    """
    for element in x:
        for new in element: 
            temp.append(normalize(new))
            print temp
            np.append(X,temp)
            temp = []
            print X
    """
    print X
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
