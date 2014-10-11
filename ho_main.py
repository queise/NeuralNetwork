#!/usr/bin/python

import sys
import re
import numpy as np
import scipy.optimize
from ho_func import *
from ho_nnclass import *

def main():

    # The data to be analysed is read from csv file and stored in a dictionary:
    ldict = F_createdictfromcsv('diabetic_data.csv')

    # Characteristics of the neural network (more on ho_nnclass):
    L = 5                                 # total number of layers (1 input, 1 output, (L-2) hidden )
    lambdas = [ 0., 0.01, 0.1, 1., 10. ]     	# regularization parameter
    K = 2                                 # num of units of the output layer

    # Creation of input and output matrixes:
    X, y = F_createXymatrixes(ldict, K, fraction=0.1)

    # Initializes Neural Network:
    NN = NeuralNetwork(X, y, L, lambdas)

    # Print information about the NN:
    NN.F_printnninfo()

    # Check correctness of gradient calculation:
#    NN.F_checkGradient()	# very slow, comment once tested

    # Training and testing network:
    NN.F_traintest()

    # Prints results:
    NN.F_printnnresults()   

    # Evaluates the neural network by calculating the learning curves:
    F_calcLearingCurve(ldict, K, NN)


if __name__ == '__main__':
  main()
