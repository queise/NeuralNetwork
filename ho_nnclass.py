import sys
import numpy as np
from itertools import groupby
from joblib import Parallel, delayed
import multiprocessing
from ho_func import *
from ho_nnfunc import *

class NeuralNetwork(object):
    def __init__(self, X, y, L, lambdas):
        # Full input data:
        self.X          = X
        self.Xbias      = np.r_[ np.ones((1, np.shape(X)[0] )), X.T]
        # Full output data:
        self.y          = y
        self.yk         = y.T
        # Training set:
        self.m_train    = int(0.7*(np.shape(X)[0]))
        self.X_train, self.Xbias_train, self.y_train, self.yk_train = self.F_createfromXymats(0,self.m_train)
        # Test set:
        self.m_test     = int(0.3*(np.shape(X)[0]))
        self.X_test, self.Xbias_test, self.y_test, self.yk_test     = self.F_createfromXymats(self.m_train,(self.m_train+self.m_test))
        # Number of layers of network:
        self.L          = L
        # Number of units in input layer:
        self.s1         = np.shape(X)[1]
        # Number of units in output layer:
        self.K          = np.shape(y)[1]
        # Number of units in each layer:
        self.sl         = self.F_createslarray() 		# all hidden layers have sl=s1
        # Regularization parameter(s):
        self.lams       = lambdas                     		# an array, to try different values if needed
        # Parameters thetas:
        self.thetas     = self.F_createthetasmatrixlist()  	# a list with L elements, the first (thetasmatrixlist[0]=None) is useless, and each other element is a matrix of dimension s_(i+1) x (s_i + 1)
        self.params     = self.F_allthetasinarow()
        self.params_opt = [None]*len(lambdas)			# here we will store the optimized parameters (after training)
        # Initialize cost row(s):
        self.Js_train   = np.array(np.zeros(len(lambdas)))
        self.Js_test    = np.array(np.zeros(len(lambdas)))
        # Initialize accuracy:
        self.Accs_train = np.array(np.zeros(len(lambdas)))
        self.Accs_test  = np.array(np.zeros(len(lambdas)))
        # The best predictor is for the index (corresponding to the lambdas array)
        self.indexofmaxaccuracy = 0

    def F_traintest(self):
        """ Trains the neural network with the training set, and then tests it with the test sets.
            For the training (optimization of parameters) uses scipy.optimize.fmin_cg().
            Both training and testing functions are called through F_testtrainsingle() (function in ho_nnfunc),
            for a simpler implementation of the parallellization. """
        num_cores = multiprocessing.cpu_count()
        if len(self.lams)>1: print '\n *** The optimization of parameters will run in parallel for different lambdas.\n'
        tupleout = Parallel(n_jobs=num_cores)(delayed(F_testtrainsingle)
                             (ind,lamda, self.params, self.sl, self.X_train, self.y_train, self.yk_train, self.Xbias_train, self.m_train,
                             self.X_test, self.y_test, self.yk_test, self.Xbias_test, self.m_test) for ind,lamda in enumerate(self.lams) )
        for ind,out in enumerate(tupleout): # each element of the tupleout consists of 5 quantities which are now stored
            self.params_opt[ind], self.Js_train[ind], self.Js_test[ind], self.Accs_train[ind], self.Accs_test[ind] = out
        # Finds the lamda that gave the best accuracy of the predictor
        self.indexofmaxaccuracy = np.argmax(self.Accs_test)

    def F_printnnresults(self):
        # Writing results in screen and in file: 
        f = open('lam.Jtra.Acc.Jte.Acc.dat', 'a')
        for lamda, Jtrain, Acc1, Jtest, Acc2 in zip(self.lams, self.Js_train, self.Accs_train, self.Js_test, self.Accs_test):
            print 'lamda = %.3f ; Jtrain = %.2f , Acctrain = %d %% ; Jtest = %.2f , Acctest = %d %%' % (lamda, Jtrain, Acc1, Jtest, Acc2)      
            f.write("%.3f %.2f %d %.2f %d\n" % (lamda, Jtrain, Acc1, Jtest, Acc2))
        f.close()
        # Prints the parameters thetas for the lamda that gave the maximum accuracy:
        print '\n *** The best accuracy of the predictor was for lambda = %.3f:  %d %%' % (self.lams[self.indexofmaxaccuracy], self.Accs_test[self.indexofmaxaccuracy])
        print '\n *** The optimal parameters are:'
        thetas = F_paramUnroll(self.params_opt[self.indexofmaxaccuracy], self.sl)
        for ind in xrange(1,len(thetas)):
            print 'theta',ind,': ',np.shape(thetas[ind]),'\n',thetas[ind]

    def F_printnninfo(self):
        print '\n\n *** The neural network has %i layers' % self.L
        print ' *** with the following units per layer:', self.sl[1:]
        print '\n *** The training set has %d entries.' % np.shape(self.X_train)[0]
        print '\n X_train:',np.shape(self.X_train),'\n',self.X_train
        print '\n y_train:',np.shape(self.y_train),'\n',self.y_train
        print '\n *** The test set has %d entries.' % np.shape(self.X_test)[0]
        print '\n X_test:',np.shape(self.X_test),'\n',self.X_test
        print '\n y_test:',np.shape(self.y_test),'\n',self.y_test

    def F_allthetasinarow(self):
    """ Creates the 'params' array, which consists of all the elements in the matrixes thetas together. """
        # note that self.thetas[0] is avoided because is a useless None
        if self.L==3:
            params = np.r_[self.thetas[1].T.flatten(), self.thetas[2].T.flatten()]
        elif self.L==4:
            params = np.r_[self.thetas[1].T.flatten(), self.thetas[2].T.flatten(), self.thetas[3].T.flatten()]
        elif self.L==5:
            params = np.r_[self.thetas[2].T.flatten(), self.thetas[2].T.flatten(), self.thetas[3].T.flatten(), self.thetas[4].T.flatten()]
        else:
            print "ERROR: the network architecture cannot have %i layers, sorry" % self.L
            sys.exit()
        return params

    def F_createslarray(self):
        """ Creates an array with s_1, s_2, ..., s_(L-1), s_L, with s_1 pre-defined, s_L=K also predefined,
            and all the layers in between (the hidden layers) with s_l=s1 """
        slarray = [ None ] * (self.L+1)    	# first tem slarray[0] will be kept empty (None)
        slarray[1] = self.s1       		# the array with the number of units of layer l (sl) starts with s1
        for ind in xrange(2,self.L) : slarray[ind] = self.s1   # all hidden layers have sl=s1
        slarray[self.L] = self.K        # the output layer has a specific number of units
        return slarray


    def F_createthetasmatrixlist(self):
        """ Creates a list with (L-1) elements, each element is a matrix of dimension s_(i+1) x (s_i + 1),
            filled with random small numbers through function F_randInitializeWeights() """
        assert self.L == (len(self.sl)-1), "ERROR, len(slarray)-1 must be equal to L"
        thetasmatrixlist = [ None ] * self.L       # a list with L elements, the first (thetasmatrixlist[0]=None) is useless,
                                        # and each other element is a matrix of dimension s_(i+1) x (s_i + 1)
        for ind in xrange(1,self.L):       # from 1 to L-1
            thetasmatrixlist[ind] = F_randInitializeWeights(self.sl[ind],self.sl[ind+1])
        return thetasmatrixlist


    def F_createfromXymats(self,ind_i,ind_f):
        """ Given the X and y matrixes, creates new ones which are a subset of them (e.g. for train, test sets...),
            and the bias units """
        X_mat     = self.X[ind_i:ind_f,]
        Xbias_mat = np.r_[ np.ones((1, np.shape(X_mat)[0] )), X_mat.T]
        y_mat     = self.y[ind_i:ind_f,]
        yk_mat    = y_mat.T
        return X_mat, Xbias_mat, y_mat, yk_mat


    def F_checkGradient(self):
        """ Compares the gradient computed by backpropagation with that computed numerically by finite differences.
            The average difference between both methods should be less than a threshold, otherwise the code stops. """
        gradient = F_computeGradient(self.params, self.sl, self.X_train, self.y_train, self.lams[0], Flatten=False)
        numgrad  = F_computeNumericalGradient( self.params, self.sl, self.X_train, self.y_train, self.lams[0] )
        diffvect = gradient.T - numgrad
        diffnum  = sum(np.abs(diffvect[0]))/len(diffvect[0])
        print '\n *** Differences between gradient computed and numerical:\n',diffnum
        if diffnum > 1.e-6:
            print '\n\n ATTENTION: difference between numerical gradient and function is too large, diff=',diffnum 
            print '\n\n gradient from F_computeGradient:\n',gradient
            print 'gradient from F_computeNumericalGradient:\n',numgrad
            sys.exit()


def F_calcLearingCurve(ldict, K, NN):
    """ Calculates the cost (J) for the train and test sets for different sizes of the training set, creating
        a new neural network for each case. The result is written in the LearningCurves.dat file. """
    fractionsoffulldataset = [ 0.0001, 0.001, 0.01, 0.1 , 1. ]
    lambdas = [ None ]
    lambdas[0] = NN.lams[NN.indexofmaxaccuracy]     # an array with only one value, the lambda that produced the best accuracy of the predictor
    f = open('LearningCurves.dat', 'a')
    for fraction in fractionsoffulldataset:
        print '\n ... computing learning curves (Jtrain and Jtest) for a fraction %f of the full data set' % fraction
        X, y = F_createXymatrixes(ldict, K, fraction=fraction)
        NNn  = NeuralNetwork(X, y, NN.L, lambdas)      # a new neural network with a fraction of the size of the original 
        NNn.F_traintest()
        Jtra = NNn.Js_train[0]                      # NNn.Js_train is an array with only one value (because only one lambda)
        Jtes = NNn.Js_test[0]                       # NNn.Js_test is an array with only one value (because only one lambda)
        f.write("%d %.3f %.3f\n" % (NNn.m_train, Jtra, Jtes))
    f.close()
