import import_files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from regressionalgorithms import * 
import dataloader
from dataloader import load_ctscan

def l2err(prediction,ytest):
    """ l2 error (i.e., root-mean-squared-error) """
    return np.linalg.norm(np.subtract(prediction,ytest)) / ytest.shape[0]

#Load the data
error_list = []
for i in range(3):
    print("Run=",i+1)
    # We load different samples from the datasets.
    train,test = load_ctscan(20000,10000)
    X_train,y_train = train
    X_test, y_test = test

    # Learn on the features that don't give singular matrix.
    linreg = FSLinearRegression({'features':range(69)})
    try:
        linreg.learn(X_train,y_train)
    except np.linalg.linalg.LinAlgError:
        continue
    y_pred = linreg.predict(X_test)
    #Record error
    error_list.append(l2err(y_pred,y_test))
    plt.plot(linreg.params['features'], linreg.weights, label = "Run={}".format(i+1))

# Plot graphs.
plt.xlabel("Features")
plt.ylabel("Weights")
plt.legend(loc = "upper right")
plt.title("Features vs Weights")
plt.show()
# Report mean error and standard error.
print("Mean Error=",np.sum(error_list)/len(error_list))
print("Standard Error=",np.std(error_list)/np.sqrt(len(error_list)))
