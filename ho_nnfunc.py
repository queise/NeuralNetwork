import sys
import csv
import re
import time
import numpy as np
import scipy.special
from ho_func import *


def F_computeGradient(params, slarray, X, y, lamda, yk = None, X_bias = None, Flatten = True ):
    """ The gradients (= the derivatives of the cost function over the parameters theta) are computed
        using backpropagation. This method consists in computing 'sigma[l]', which is the error of each
        node in layer l. Is computed from layer L (output) backwards. """
    K = slarray[-1]
    m, n = np.shape( X )
    L = len(slarray)-1

    thetas = F_paramUnroll(params, slarray)  # gives thetas[0] = None, thetas[1] = theta1, thetas[2]= theta2, ...
    units_a, poli_z = F_feedForward(thetas, L, X, X_bias) # gives units_a[0]=None and units_a[1]=a1,..., poli_z[0] and [1] = None , and poli_z[2]=z2,...

    if yk is None:
        yk = y.T
        assert np.shape(yk) == np.shape(units_a[L]), "Error, shape of recoded y is different from aL"

    # backpropagation
    sigma = [ None ] * (L+1) ; gradaccum = [ None ] * (L+1)
    sigma[L] = units_a[L] - yk
    for ind in xrange((L-1),1,-1):	# from L-1 to 2
        sigma[ind] = thetas[ind].T.dot( sigma[ind+1] ) * F_sigmoidGradient( np.r_[np.ones((1, m)), poli_z[ind] ] )
        sigma[ind] = sigma[ind][1:,:]

    gradient = []
    for ind in xrange(1,L):		# from 1 to L
        gradaccum[ind]       = sigma[ind+1].dot(  units_a[ind].T ) / m
        gradaccum[ind][:,1:] = gradaccum[ind][:,1:] + (thetas[ind][:,1:] * lamda / m)
        gradient = gradient + gradaccum[ind].T.reshape(-1).tolist()
    gradient = np.array([gradient]).T

    if Flatten: gradient = np.ndarray.flatten(gradient)

    return gradient


def F_feedForward(thetas, L, X, X_bias = None):
    """ Starting from the parameters thetas, this function calculates the activation units_a and the
        polynomials poli_z (linear functions of theta*unit_a), from l=2 to L. """
    one_rows = np.ones((1, np.shape(X)[0] ))

    units_a = [ None] * (L+1)
    poli_z  = [ None] * (L+1)
    units_a[1] = np.r_[one_rows, X.T]  if X_bias is None else X_bias  # a1

    for ind in xrange(2,L+1):	# from 2 to L
        poli_z[ind]  = thetas[ind-1].dot( units_a[ind-1] ) 		# z2 = theta1*a1, z3 = ...
        units_a[ind] = F_sigmoid(poli_z[ind])				# a2 = g(z2), a3 = ...
        if ind != L: units_a[ind] = np.r_[one_rows, units_a[ind]] 	# adds bias row to ai except for last a

    return units_a, poli_z


def F_computeCost(params, slarray, X, y, lamda, yk = None, X_bias = None):
    """ Computes the cost function for logistic regression with regularization.
        It is a measure of how far is the prediction of the neural network from the real outputs. """
    K 	 = slarray[-1]
    L	 = len(slarray)-1
    m, n = np.shape( X )

    thetas = F_paramUnroll( params, slarray )  # gives thetas[0] = None, thetas[1] = theta1, thetas[2]= theta2, ...
    units_a, poli_z = F_feedForward( thetas, L, X, X_bias ) # gives units_a[0]=None and units_a[1]=a1,..., poli_z[0] and [1] = None ,

    if yk is None:
        yk = y.T
        assert np.shape(yk) == np.shape(units_a[L]), "Error, shape of yK is different from aL"

    term1	= -yk * np.log( units_a[L] )
    term2 	= (1 - yk) * np.log( 1 - units_a[L] )
    left_term 	= np.sum(term1 - term2) / m

    right_term = np.sum(thetas[1][:,1:] ** 2)
    for ind in xrange(2,L):	# from 2 to L-1
        right_term = right_term + np.sum(thetas[ind][:,1:] ** 2)
    right_term = right_term * lamda / (2 * m)
    cost = left_term + right_term

    return cost


def F_computeNumericalGradient(params, slarray, X, y, lamda):
    """ Here the gradients (= the derivatives of the cost function over the parameters theta) are computed
        with finite differences (deriv(x)=f(x+e)-f(x-e)/2e) instead of by backpropagation as is normally done. """
    numgrad 	= np.zeros( np.shape(params) )
    perturb 	= np.zeros( np.shape(params) )
    e = 1e-4

    num_elements = np.shape(params)[0]
    yk = y.T

    for p in range(0, num_elements) :
        perturb[p] = e
        loss1 = F_computeCost( params - perturb, slarray, X, y, lamda, yk )
        loss2 = F_computeCost( params + perturb, slarray, X, y, lamda, yk )
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0

    return numgrad


def F_predict(Xm, thetas, L):
    """ Calculates the output of the neural network for given inputs and optimized (trained) parameters thetas.
        The real outputs would be units_a[L], but as this is a classification problem we just look for the
        index of the element with a greater value. """
# similar to function feedForward
    units_a = [ None] * (L+1)
    poli_z = [ None] * (L+1)
    units_a[1] = np.r_[np.ones((1, 1)), Xm.reshape( np.shape(Xm)[0], 1 )]	# a1

    for ind in xrange(2,L+1):     # from 2 to L
        poli_z[ind]  = thetas[ind-1].dot( units_a[ind-1] )          # z2 = theta1*a1, z3 = ...
        units_a[ind] = F_sigmoid(poli_z[ind])                               # a2 = g(z2), a3 = ...
        if ind != L: units_a[ind] = np.r_[np.ones((1, 1)), units_a[ind]]   # adds bias row to ai except for last a
#    print 'poli_z[',L,']:',poli_z[L]
#    print 'units_a[',L,']:',units_a[L]
    return np.argmax(poli_z[L])


def F_computeAccuracy(params, slarray, m, X, y):
    """ The accuracy is computed comparing the prediction of the trained network with the actual values
        of the outputs. """
    L       = len(slarray)-1
    thetas = F_paramUnroll( params, slarray )  # gives thetas[0] = None, thetas[1] = theta1, thetas[2]= theta2, ...
    counter = 0
#  f2 = open('pred.act.dat', 'a')
    for i in range(0,m):
        prediction = F_predict( X[i], thetas, L )
        actual = np.argmax( y[i] )
#        print 'prediction=',prediction,' actual=',actual
#        f2.write("p%f a%f\n" % (prediction, actual))
        if( prediction == actual ):
            counter+=1
    Accur = counter * 100 / m
#    f2.close()  
    return Accur

def F_testtrainsingle(ind, lamda, params, sl, X_train, y_train, yk_train, Xbias_train, m_train,
                      X_test,  y_test, yk_test, Xbias_test, m_test):
    """ Trains the neural network with the training set, and then tests it with the test sets.
        For the training (optimization of parameters) uses scipy.optimize.fmin_cg().
        Returns the optimized parameters thetas (in a row instead of original matrixes) and the
        cost functions and accuracies of the training and test sets. """
    print '\n ... training neural network for lambda=%.4f (it may take a while)' % lamda
    params_opt = scipy.optimize.fmin_cg( F_computeCost, x0=params, fprime=F_computeGradient, \
                                          args=( sl, X_train, y_train, lamda, yk_train, Xbias_train), \
                                          maxiter=1000, disp=True )  # with full_output=True take output [0]
    Js_train   = F_computeCost(params_opt, sl, X_train, y_train, lamda, yk_train, Xbias_train)
    Js_test    = F_computeCost(params_opt, sl, X_test, y_test, lamda, yk_test, Xbias_test)
    Accs_train = F_computeAccuracy(params_opt, sl, m_train, X_train, y_train)
    Accs_test  = F_computeAccuracy(params_opt, sl, m_test, X_test, y_test)
    return params_opt, Js_train, Js_test, Accs_train, Accs_test
