import sys
import csv
import re
import time
import numpy as np
import scipy.special
from joblib import Parallel, delayed
import multiprocessing


def F_createdictfromcsv(file):
    """ Returns a dictionary from the csv file, each entry is a line, the keys are indexes (0,...,numlines-1)
        and the content of each entry is a row with all the info (columns) in the csv file. """
    jdict = {}
    with open(file, 'rb') as csvfile:
        reader = csv.reader(csvfile)
        for ind,row in enumerate(reader):
            jdict[ind] = row[0:]	# the key is an index
    return jdict

def F_msets(m_total, train_frac, test_frac, CV_frac):
    """ From the total number of entries in the data set (m_total) and specified fractions for
        the training, test and CV sets, returns the number of entries for each set. """
#    assert train_frac + test_frac + CV_frac == 1, "ERROR, train+test+CV fractions must sum 1, there are %f, %f and %f" % (train_frac,test_frac,CV_frac)
    m_train = int( train_frac * m_total )                            			# number of training set
    m_test  = int( (train_frac+test_frac)*m_total ) - int( train_frac*m_total )           # test set
    m_CV    = m_total - int( (train_frac+test_frac)*m_total )    				# cross-validation set
    assert m_train + m_test + m_CV == m_total, "ERROR, m_train+m_test+m_CV must sum m_total"
    return m_train, m_test, m_CV


def F_countdiag(ldict,m,listsetdiag,ndiffdiag):
    """ Uses the 3 diagnostic codes in each entry of the data set (ldict[row_ind][18:21])
        to count the ocurrences of each diagnostic.
        Returns a dictionary: the keys are the diagnostic codes, the values are the number of ocurrences. """
    diagcount = {}
    for row_ind in xrange(0,m):
        for diag in ldict[row_ind][18:21]:
            if not diag in diagcount: # Special case if we're seeing this diagnostic code for the first time.
                diagcount[diag] = 1
            else:
                diagcount[diag] = diagcount[diag] + 1
    return diagcount

def F_createXymatrixes(ldict, K, fraction=1):
    """ Returns the input (X) and output matrixes (y), given the dictionary (ldict) with all the data set.
        The number of ouputs (K, classes) can be specified to be either 2 or 3.
        Only a fraction of the data base can be choosen to speed up the computation (data set has 10^5 entries),
        or for evaluation purposes. """
    m = int(len(ldict)*fraction)	# number of entries in the fraction of the data base used
    # The fraction of the full database will be randomly choosen among the entries:
    randomlist_m_indexes = np.random.permutation(m)
    # There are more than 900 different diagnostics in the data set. To speed up the computation,
    # we may only want to use the more used ones as inputs.
    nselectdiag = 100   # the top nselectdiag will be the only ones used as inputs of the neural network (or less if not enough different diagnostics in the fraction of the dataset choosen)
    listdiag  = []
    for row_ind in randomlist_m_indexes:
        if ldict[row_ind][18] != '?': listdiag.append(ldict[row_ind][18])
        if ldict[row_ind][19] != '?': listdiag.append(ldict[row_ind][19])
        if ldict[row_ind][20] != '?': listdiag.append(ldict[row_ind][20])
    listsetdiag = list(set(listdiag))
    ndiffdiag = len(listsetdiag)	# number of different diagnostic codes in columns 18,19 and 20 of data dict (for this fraction of dataset)
    diagcount = F_countdiag(ldict,m,listsetdiag,ndiffdiag)	# creates a dictionary: diag code is the key, value is the number of ocurrences
    nselectdiag = min(nselectdiag,ndiffdiag)
    topdiag = sorted(diagcount, key=diagcount.get, reverse=True)[:nselectdiag+1]
    
    ndrugs    = 24				# the administered drugs is a particular type of input that is implemented in a for loop
    s1        = 15 + ndrugs + nselectdiag	# total number of inputs
    mat_X     = np.zeros((m,s1))
    mat_y     = np.zeros((m,K), dtype=np.int8)

    for row_ind in randomlist_m_indexes:
        # input 1: is male?
        mat_X[row_ind,0] = 1 if (ldict[row_ind][3]=='Male') else 0
        # input 2: is female?
        mat_X[row_ind,1] = 1 if (ldict[row_ind][3]=='Female') else 0
        # input 3: age
        str = ldict[row_ind][4]
        tuple = re.search(r'\[(\d+)-(\d+)\)',str)
        age = (int(tuple.group(2))+int(tuple.group(1)))/2 # the age is calculated a the mean of the interval
        mat_X[row_ind,2] = age
        # input 4: time in hospital in days
        mat_X[row_ind,3] = ldict[row_ind][9]
        # input 5: Number of lab tests performed during the encounter
        mat_X[row_ind,4] = ldict[row_ind][12]
        # input 6: Number of procedures (other than lab tests) performed during the encounter
        mat_X[row_ind,5] = ldict[row_ind][13]
        # input 7: Number of distinct generic names (of medications) administered during the encounter 
        mat_X[row_ind,6] = ldict[row_ind][14]
        # input 8: Number of outpatient visits of the patient in the year preceding the encounter 
        mat_X[row_ind,7] = ldict[row_ind][15]
        # input 9: Number of emergency visits of the patient in the year preceding the encounter
        mat_X[row_ind,8] = ldict[row_ind][16]
        # input 10: Number of inpatient visits of the patient in the year preceding the encounter
        mat_X[row_ind,9] = ldict[row_ind][17]
        # Primary+secondary+additional diagnostic (ndiffdiag~916 different classification values, only the top nselectdiag are used)
        for dind in xrange(1,nselectdiag+1):
            mat_X[row_ind,9+dind] = 1 if (ldict[row_ind][18] == topdiag[dind-1] or ldict[row_ind][19] == topdiag[dind-1] or 
                                          ldict[row_ind][20] == topdiag[dind-1]) else 0
        # Number of diagnoses entered to the system
        mat_X[row_ind,10+nselectdiag] = ldict[row_ind][21]
        # Glucose serum test result (value: >200, >300, normal, and none)
        mat_X[row_ind,11+nselectdiag] = 1 if (ldict[row_ind][22] == '>300') else 0
        # A1c test result (values: >8, >7, normal and none)
        mat_X[row_ind,12+nselectdiag] = 1 if (ldict[row_ind][23] == '>8') else 0
        # Indicates if there was a change in diabetic medications
        mat_X[row_ind,13+nselectdiag] = 1 if (ldict[row_ind][47] == 'Ch') else 0
        # Indicates if there was any diabetic medication prescribed
        mat_X[row_ind,14+nselectdiag] = 1 if (ldict[row_ind][48] == 'Yes') else 0
        # inputs of 24 drugs:
        for cind in xrange(0,ndrugs):
            mat_X[row_ind,15+nselectdiag+cind] = 0 if (ldict[row_ind][24+cind] == 'No') else 1
#            mat_X[row_ind,15+nselectdiag+cind] = 1 if (ldict[row_ind][24+cind] == 'Up') else 0

        # output:
        if K==3:
            if ldict[row_ind][49] == 'NO':
                mat_y[row_ind,0:3] = [ 0, 0, 1 ]
            elif ldict[row_ind][49] == '>30':
                mat_y[row_ind,0:3] = [ 0, 1, 0 ]
            elif ldict[row_ind][49] == '<30':
                mat_y[row_ind,0:3] = [ 1, 0, 0 ]
            else:
                print 'ERROR: data value %s not expected' % ldict[row_ind][49]
                sys.exit()
        elif K==2:
            if ldict[row_ind][49] == '<30': # or ldict[row_ind][49] == '>30':
                mat_y[row_ind,0:2] = [ 1, 0 ]
            else:
                mat_y[row_ind,0:2] = [ 0, 1 ]
        else:
            print 'ERROR: only 2 or 3 outputs are possible, not K=',K
            sys.exit() 

    # the order of the set is randomized: (not necessary anymore, as the randomization is done previously)
#    mat_X, mat_y = F_shuffle_in_unison_inplace(mat_X,mat_y)

    return mat_X, mat_y


def F_shuffle_in_unison_inplace(a, b):
    """ Randomizes the order of two arrays accordingly """
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def F_sigmoid(z):
    """ The logistic function, defined as expit(z) = 1/(1+exp(-z)) """
    return scipy.special.expit(z)


def F_sigmoidGradient(z):
    """ Function used for backpropagation. """
    sig = F_sigmoid(z)
    return sig * (1 - sig)


def F_allthetasinacolumn(L, thetas):
    """ Creates the 'params' array, which consists of all the elements in the matrixes thetas together. """
    if L==3:
        params = np.array([thetas[1].T.reshape(-1).tolist() + thetas[2].T.reshape(-1).tolist()]).T
    elif L==4:
        params = np.array([thetas[1].T.reshape(-1).tolist() + thetas[2].T.reshape(-1).tolist() + 
                           thetas[3].T.reshape(-1).tolist()]).T
    elif L==5:
        params = np.array([thetas[1].T.reshape(-1).tolist() + thetas[2].T.reshape(-1).tolist() + 
                           thetas[3].T.reshape(-1).tolist() + thetas[4].T.reshape(-1).tolist()]).T
    elif L==6:
        params = np.array([thetas[1].T.reshape(-1).tolist() + thetas[2].T.reshape(-1).tolist() + 
                           thetas[3].T.reshape(-1).tolist() + thetas[4].T.reshape(-1).tolist() + 
                           thetas[5].T.reshape(-1).tolist()]).T 
    else:
        print "ERROR: the network architecture cannot have %i layers, sorry" % L
        sys.exit()
    return params


def F_randInitializeWeights(L_in, L_out):
    """ Creates a matrix with dimensions (L_out, L_in + 1) with small random positive and negative numbers. """
    e = np.sqrt(6.)/(np.sqrt(L_in+L_out))
    w = np.random.random((L_out, L_in + 1)) * 2 * e - e
    return w

def F_paramUnroll( params, slarray ):
    """ Does the opposite than function F_allthetasinacolumn. With the 'slarray' the function knows
        the original sizes of the thetas matrixes, and then it can rebuild them from the params array,
        which consists of all elements of the thetas matrixes in a single array. """
    L = len(slarray)-1
    theta_elems = [ 0 ] * L
    thetas = [ None ] * L
    theta_size = [ None ] * L
    for ind in xrange(1,L):       # from 1 to (L-1)
        theta_elems[ind] = (slarray[ind] + 1) * slarray[ind+1]
        theta_size[ind]  = (slarray[ind] + 1 , slarray[ind+1] )
        thetas[ind]      = params[theta_elems[ind-1]:(theta_elems[ind-1]+theta_elems[ind])].T.reshape( theta_size[ind] ).T
    return thetas
