import import_files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from regressionalgorithms import * 
from dataloader import load_ctscan
from dataloader import splitdataset

# Calculate l2err.
def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))
# Calculate l2 norm.
def l2(vec):
    """ l2 norm on a vector """
    return np.linalg.norm(vec)

#Load the data
train,test = load_ctscan(20000,10000)
X_train, y_train = train
X_test, y_test = test        

"""
Defines the object for performing Stocastic Gradient Descent.
learn function performs SGD using training and test.
"""
class StocasticLinearRegression():
    def __init__(self):
        #self.num_feature = num_features
        self.reg = FSLinearRegression({'features':list(range(385))})
        self.reg.weights = np.array([])
    # Learn weights using SGD
    def learn(self,X_train,y_train,epochs):
        self.reg.weights = np.zeros(X_train.shape[1])
        alpha = 0.01
        for _ in range(epochs):
            # Shuffle the data.
            combine = np.column_stack([X_train,y_train])
            shuffle = np.random.permutation(combine)
            X_train,y_train = shuffle[:,:385],shuffle[:,385]
            for idx in list(range(X_train.shape[0])):
                alpha = alpha/(idx+1)
                y_pred = np.dot(X_train[idx],self.reg.weights)
                gradient = np.dot((y_pred - y_train[idx]),X_train[idx])
                self.reg.weights = self.reg.weights - alpha * gradient

# Run the SGD algorithm using multiple epochs and plot the graph.
def run():
    error_list = []
    epochs = list(range(100))
    t = []
    for i in epochs:
        starttime = time.time()
        stocastic_regression = StocasticLinearRegression()
        stocastic_regression.learn(X_train,y_train,i)
        y_pred = stocastic_regression.reg.predict(X_test)
        error_list.append(l2err(y_pred,y_test)/y_test.shape[0])
        endtime = time.time()
        t.append((endtime - starttime))
        
    plt.plot(epochs,error_list)
    plt.title("SGD epochs vs error")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.show()

    plt.plot(error_list,t)
    plt.title("SGD error vs Runtime")
    plt.xlabel("Error")
    plt.ylabel("Runtime")
    plt.show()

run()
