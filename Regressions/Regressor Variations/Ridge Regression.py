# Contains import that are system specefic. (basically I feel cleaner with
# no import sys). 
import import_files

import numpy as np

from regressionalgorithms import * 
import dataloader
from dataloader import load_ctscan

# Calculate L2 error.
def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))

#Load the data
train,test = load_ctscan(20000,10000)
X_train,y_train = train
X_test, y_test = test


# Run the regression algorithm on the features which
# will not give a singular matrix.

# Learn the weights using ridge regression. The code for ridge regression
# can be found in code/regressionalgorithms.py, the ridge regression is
# implemented in closed form by adding bias inside the inverse function.

def learn(param_name,values):
    for meta_param in values:
        linreg = RidgeRegression({'features':range(385)})
        try:
            linreg.learn(X_train,y_train, meta_param)
        except np.linalg.linalg.LinAlgError:
            continue
        y_pred = linreg.predict(X_test)
        test_error = l2err(y_pred, y_test)/y_test.shape[0]
        y_train_pred = linreg.predict(X_train)
        train_error = l2err(y_train_pred, y_train)/y_train.shape[0]
        print("{} = ".format(param_name),meta_param,"Train Error = ",train_error)
        print("{} = ".format(param_name),meta_param,"Test Error = ",test_error)

learn("lambda",[0.01,0.1,1.0])
