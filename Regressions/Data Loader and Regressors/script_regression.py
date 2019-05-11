from __future__ import division  # floating point division
import csv
import random
import math
import numpy as np

import dataloader as dtl
import regressionalgorithms as algs
import plotfcns

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

def l1err(prediction,ytest):
    """ l1 error """
    return np.linalg.norm(np.subtract(prediction,ytest),ord=1) 

def l2err_squared(prediction,ytest):
    """ l2 error squared """
    return np.square(np.linalg.norm(np.subtract(prediction,ytest)))

def geterror(predictions, ytest):
    # Can change this to other error values
    return l2err(predictions,ytest)/ytest.shape[0]


if __name__ == '__main__':
    trainsize = 1000
    testsize = 5000
    numparams = 1
    numruns = 1
    
    regressionalgs = {'Random': algs.Regressor(),
                'Mean': algs.MeanPredictor(),
                'FSLinearRegression5': algs.FSLinearRegression({'features': [1,2,3,4,5]}),
                'FSLinearRegression50': algs.FSLinearRegression({'features': range(50)})
             }       
    numalgs = len(regressionalgs)

    errors = {}
    for learnername in regressionalgs:
        errors[learnername] = np.zeros((numparams,numruns))
        
    trainset, testset = dtl.load_ctscan(trainsize,testsize)
    print('Running on train={0} and test={1}').format(trainset[0].shape[0], testset[0].shape[0])

    # Currently only using 1 parameter setting (the default) and 1 run
    p = 0
    r = 0
    params = {}
    for learnername, learner in regressionalgs.iteritems():
    	# Reset learner, and give new parameters; currently no parameters to specify
    	learner.reset(params)
    	print('Running learner = ' + learnername + ' on parameters ' + str(learner.getparams()))
    	# Train model
    	learner.learn(trainset[0], trainset[1])
    	# Test model
    	predictions = learner.predict(testset[0])
    	error = geterror(testset[1], predictions)
    	print('Error for ' + learnername + ': ' + str(error))
    	errors[learnername][p,r] = error
                
                

