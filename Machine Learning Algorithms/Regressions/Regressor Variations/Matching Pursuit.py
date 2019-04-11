import import_files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regressionalgorithms import * 
from dataloader import load_ctscan
from dataloader import splitdataset

#Calculate l2 Error.
def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

#Calculate l2 norm.
def l2(vec):
    """ l2 norm on a vector """
    return np.linalg.norm(vec)

#Load the data
train,test = load_ctscan(20000,10000)
X_train, y_train = train
X_test, y_test = test        
error = []
feat = []

"""
Creates an object for performing Matching Pursuit.
learn function performs MP task.
"""
class MPRegression():
    def __init__(self,num_features=100):
        self.num_features = num_features
        self.reg = FSLinearRegression()
        self.selected_features = []
        self.reg.weights = np.array([])

    def learn(self,X_train,y_train,corr_epsilon = 0.3,res_epsilon=0.07):
        # I ussually prefer to say validation set when spliting data inside
        # the learn function, hence Xv and yv notation.
        Xv_train, yv_train = X_train, y_train
        
        best_idx = -1
        best_pear_corr = float('inf')
        # Initialize the residual with the training target values.
        residual = yv_train
        # A list of columns which are to be rejected. (if any, implemented to
        # safe guard from any exceptions.
        reject = []
        # Initialize to zeros. Keeps track of residual in previous iteration.
        prev_residual = np.zeros(yv_train.shape[0])
        
        while ((best_pear_corr > corr_epsilon) and ((np.abs(l2(residual) - l2(prev_residual)))/prev_residual.shape[0] > res_epsilon)  and (len(self.selected_features) < self.num_features)):
            best_pear_corr = -1
            for idx in sorted(list(range(Xv_train.shape[1]))):
                # select index and create a weight column.
                if idx not in self.selected_features and idx not in reject:
                    s = self.selected_features + [idx]
                    w = np.append(self.reg.weights, np.array([0]),axis=0)
                else:
                    continue
                normalizer = np.sqrt(np.sum(residual**2) * np.sum(Xv_train[:,idx]**2) )
                # Calculate pearson's correlation (normalized)
                pear_corr = np.abs(np.dot(Xv_train[:,idx].T,residual))/ normalizer
                # record the best correlation so far.
                if pear_corr > best_pear_corr:
                    best_idx = idx
                    best_pear_corr = pear_corr
                    
            self.selected_features.append(best_idx)
            self.reg.params = {'features':self.selected_features}
            
            try:
                self.reg.learn(Xv_train,yv_train)
            except np.linalg.linalg.LinAlgError:
                reject.append(self.selected_features.pop())
            prev_residual = residual
            # Calculate new residuals.
            residual =  np.dot(Xv_train[:,self.selected_features]
                                         ,self.reg.weights) - yv_train
    
    def predict(self,X_test):
        Xless = X_test[:,self.selected_features]
        return np.dot(Xless,self.reg.weights)
            
mpreg = MPRegression()
mpreg.learn(X_train,y_train)
y_pred = mpreg.predict(X_test)

print("Total number of features selected =", len(mpreg.selected_features))
print("Features selected =", mpreg.selected_features)
print("Error =", l2err(y_pred,y_test)/y_test.shape[0])
