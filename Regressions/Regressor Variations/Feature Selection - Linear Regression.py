import import_files

import numpy as np

from regressionalgorithms import * 
import dataloader
from dataloader import load_ctscan

# Calculate L2 Error.
def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest))


#Load the data
train,test = load_ctscan(20000,10000)
X_train,y_train = train
X_test, y_test = test
print(X_train)

# Results of this script is in the main pdf report.
# This is just an interface to experiment on
# FSLinearRegression.

linreg = FSLinearRegression()
linreg.reset({'features':range(1,384)})
print(linreg.getparams())
linreg.learn(X_train,y_train)
y_pred = linreg.predict(X_test)
print(linreg.weights)
print(l2err(y_pred, y_test)/y_test.shape[0])
