import import_files

import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

from regressionalgorithms import * 
import dataloader
from dataloader import load_ctscan
from time import sleep

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))


def l2(vec):
    """ l2 norm on a vector """
    return np.linalg.norm(vec)


def loadcsv(filename):
    dataset = np.genfromtxt(filename, delimiter=',')
    return dataset

def load_ctscan(trainsize=5000, testsize=5000):
    """ A CT scan dataset """
    if trainsize + testsize < 5000:
        filename = '../code/datasets/slice_localization_data.csv'
    else:
        filename = '../code/datasets/slice_localization_data.csv'
    dataset = loadcsv(filename)
    trainset, testset = splitdataset(dataset,trainsize, testsize,featureoffset=1)    
    return trainset,testset

def rmse(vec):
    return np.sqrt(sum(vec)/vec.shape[0])

# Changed the normalizing.
def splitdataset(dataset, trainsize, testsize, testdataset=None, featureoffset=None, outputfirst=None):
    """
    Splits the dataset into a train and test split
    If there is a separate testfile, it can be specified in testfile
    If a subset of features is desired, this can be specifed with featureinds; defaults to all
    Assumes output variable is the last variable
    """
    randindices = np.random.randint(0,dataset.shape[0],trainsize+testsize)
    featureend = dataset.shape[1]-1
    outputlocation = featureend    
    if featureoffset is None:
        featureoffset = 0
    if outputfirst is not None:
        featureoffset = featureoffset + 1
        featureend = featureend + 1
        outputlocation = 0
    
    Xtrain = dataset[randindices[0:trainsize],featureoffset:featureend]
    ytrain = dataset[randindices[0:trainsize],outputlocation]
    Xtest = dataset[randindices[trainsize:trainsize+testsize],featureoffset:featureend]
    ytest = dataset[randindices[trainsize:trainsize+testsize],outputlocation]

    if testdataset is not None:
        Xtest = dataset[:,featureoffset:featureend]
        ytest = dataset[:,outputlocation]        

    # Normalize features, with maximum value in training set
    # as realistically, this would be the only possibility    
    for ii in range(Xtrain.shape[1]):
        Xtrain[:,ii] = np.divide(Xtrain[:,ii], np.sqrt(np.sum(Xtrain[:,ii]**2)))
        Xtest[:,ii] = np.divide(Xtest[:,ii], np.sqrt(np.sum(Xtrain[:,ii]**2)))
                        
    # Add a column of ones; done after to avoid modifying entire dataset
    Xtrain = np.hstack((Xtrain, np.ones((Xtrain.shape[0],1))))
    Xtest = np.hstack((Xtest, np.ones((Xtest.shape[0],1))))
                              
    return ((Xtrain,ytrain), (Xtest,ytest))


#Load the data
train,test = load_ctscan(20000,10000)
X_train,y_train = train
X_test, y_test = test

'''
This class applies lasso regularization to linear regression.

Input: lambda_param : The regularization meta parameter.
       step_size : The tolerance for changes in the residuals.
       max_iter : The maximum number of iteration to run the coordinate descent
                  if it doesn't converge.
Output: (void) Set weights for linear regression.
'''
class Lasso():
    def __init__(self, lambda_param= 0.05,step_size=0.00001, max_iter = 20):
        self.lambda_param = lambda_param
        self.reg = FSLinearRegression({'features':list(range(385))})
        self.reg.weights = None
        self.step_size = step_size
        self.max_iter = max_iter

    ## Coordinate descent to solve the lasso problem, takes in the features
    ## of the training set and the target variable.
    def coordinate_descent(self,Xv_train,yv_train):
        # self.reg.weights : sets the weight of the linear regression. see the __init__ for
        # constructor of FSLinearRegression.
        self.reg.weights = np.random.random(Xv_train.shape[1])
        
        converged = False
        residual = yv_train
        ## The loop runs till the convergence condition is met or the number
        ## of iterations are reached.
        while (~(converged)&(self.max_iter > 0)) :
            step_sizes = []
            for idx in sorted(list(range(Xv_train.shape[1]))):  # run for all the features in the training set.
                
                # find the predicted value by removing one features/coordinate at a time.
                yv_pred = np.dot(np.delete(Xv_train,idx,axis = 1), np.delete(self.reg.weights,idx,axis = 0))    
                prev_residual = residual
                residual = yv_train - yv_pred # calculate the residual.
                normalizer = np.sqrt(np.sum(residual**2) * np.sum(Xv_train[:,idx]**2) )
                rho = np.dot(residual,Xv_train[:,idx]) / normalizer # calculate the correlation with the index choosen.
    
                # Apply soft thresholding condition for each features, each control statement holds one condition.
                if rho < (-self.lambda_param/2): 
                    self.reg.weights[idx] = rho + self.lambda_param/2
                elif ((rho >= -self.lambda_param/2) and (rho <= self.lambda_param/2)):
                    self.reg.weights[idx] = 0
                elif rho > (self.lambda_param/2):
                    self.reg.weights[idx] = rho - self.lambda_param/2
                step_sizes.append(np.abs(rmse(residual) - rmse(prev_residual)))
            converged = max(step_sizes) < self.step_size # Check if converged.
            self.max_iter = self.max_iter - 1 # decrease the max_iteration.
        print("Lambda=",self.lambda_param)
        print("Number of zero weights=",sum(np.abs(self.reg.weights) == 0))

lasso = Lasso()
lasso.coordinate_descent(X_train,y_train)
y_pred = lasso.reg.predict(X_test)
print("Test error=",l2err(y_pred,y_test)/y_test.shape[0])

