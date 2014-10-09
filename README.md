neuralnetwork
=============

A **simple** python neural network to analyse data from diabetic patients in 130 US hospitals [1].
The architecture of the network is flexible (number of layers, input and output units).
The neural network classifies using regularized logistic regression. The gradients are computed with backpropagation and are checked numerically. The network is optimized with the scipy.optimize.fmin_cg algorithm. When several regularization parameters are used, the optimization is parallelized. Finally, learning curves are computed to evaluate the performance of the neural network.

- [ho_main.py](ho_main.py) :    Main structure, see the comments on the script for the details

- [ho_nnclass.py](ho_nnclass.py) :    Definition of the NeuralNetwork class and functions.

- [ho_nnfunc.py](ho_nnfunc.py) :    Functions to compute the cost, gradient, accuracy...

- [ho_func.py](ho_func.py) :  Among other functions, the ones that create the matrixes with inputs and outputs of the neural network.

- diabetic_data.csv : Not provided here, can be obtained at [this link](https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008) [2]


Dependencies: [scipy](http://www.scipy.org/), [numpy](http://www.numpy.org/), [joblib](https://pythonhosted.org/joblib/).

In the initial stage of development I used functions from this [repository](https://github.com/subokita/mlclass/blob/master/ex4.py).



[1] Beata Strack, Jonathan P. DeShazo, Chris Gennings, Juan L. Olmo, Sebastian Ventura, Krzysztof J. Cios, and John N. Clore, “Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient Records,” BioMed Research International, vol. 2014, Article ID 781670, 11 pages, 2014.

[2] https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

