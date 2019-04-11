import import_files

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import time
from regressionalgorithms import * 
from dataloader import load_ctscan
from dataloader import splitdataset

#Calculate l2 error.
def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

# Calculate l2 norm.
def l2(vec):
    """ l2 norm on a vector """
    return np.linalg.norm(vec)

# Calculate error.
def Err(w,X,y):
    return np.dot((np.dot(X,w) - y).T,(np.dot(X,w) - y)) / X.shape[0]

# Calculate gradient of the error.
def gradErr(X,w,y):
    return (2 * np.dot(np.dot(X.T,X),w)) - (2 * np.dot(X.T,y))

# Calculate alpha, the function is defined by me, I tried to incorporate
# an exponetial decrease in the value of alpha as the number of iterations
# increase. 
def diminishing_alpha(k,alpha):
    return (alpha)**(k*(1.95 - alpha))
    
#Load the data
train,test = load_ctscan(20000,10000)
X_train, y_train = train
X_test, y_test = test        

"""
Perform Batch Gradient Descent
"""
class BatchGradientLinearRegression():
    
    def __init__(self):
        #self.num_feature = num_features
        self.reg = FSLinearRegression({'features':list(range(385))})
        self.reg.weights = None

    def learn(self,X_train,y_train,epochs):
        Xv_train, yv_train = X_train, y_train
        self.reg.weights = np.zeros(Xv_train.shape[1])
        tolerance = 10e-12
        alpha = 0.001
        err = float('inf')
        k=0
        for _ in range(epochs):
            # Shuffle the data.
            combine = np.column_stack([Xv_train,yv_train])
            shuffle = np.random.permutation(combine)
            Xv_train,yv_train = shuffle[:,:385],shuffle[:,385]

            while np.abs(Err(self.reg.weights,Xv_train,yv_train) - err) > tolerance:
                err = Err(self.reg.weights,Xv_train,yv_train)
                k = k+1
                alpha =  diminishing_alpha(k,alpha)
                self.reg.weights = self.reg.weights - alpha * np.dot(X_train.T,(np.dot(X_train,self.reg.weights)-yv_train))        

# Run the Batch GD algorithm using multiple epochs and plot the graph.
def run():
    error_list = []
    epochs = list(range(100))
    t = []
    for i in epochs:
        starttime = time.time()
        batch_regression = BatchGradientLinearRegression()
        batch_regression.learn(X_train,y_train,i)
        y_pred = batch_regression.reg.predict(X_test)
        error_list.append(l2err(y_pred,y_test)/y_test.shape[0])
        endtime = time.time()
        t.append((endtime - starttime))
    plt.plot(epochs,error_list)
    plt.title("Batch GD epochs vs error")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

    plt.plot(error_list,t)
    plt.title("Batch GD epochs vs error")
    plt.xlabel("Error")
    plt.ylabel("Runtime")
    plt.show()

run()
          
