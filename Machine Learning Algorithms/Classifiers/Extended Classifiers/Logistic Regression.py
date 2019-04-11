import import_files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import load_susy
from dataloader import splitdataset

#Load the data
train,test = load_susy(2000,1000)
X_train, y_train = train
X_test, y_test = test

"""
An Implementation of logistic regression.
This class consists of
    theta = weight coefficients for logistic regression.
    eta = Learning rate.
    max_iter = maximum number of iterations.
    show_cost_graph = boolean value to view the cost vs iteration graph.
"""

class LogisticRegression():

    def __init__(self,eta = 0.005, max_iter = 100,show_cost_graph = True):
        self.theta = None
        self.eta = eta
        self.max_iter = max_iter
        self.show_cost_graph = show_cost_graph
        

    # Sigmoid function.
    def sigmoid(self,x):
        return 1/(1+ np.exp(- np.dot(self.theta,x.T)))

    # Get the cost for every iteration for checking.
    def getCost(self,X_train,y_train):
        cost = 0
        for i in range(X_train.shape[0]):
            cost += (y_train[i] * np.log(self.sigmoid(X_train[i]))) + ((1- y_train[i]) * np.log(1-self.sigmoid(X_train[i])))
        return (-cost/X_train.shape[0])
    
    # Learning the logistic regression weight coefficients using gradient descent.
    def learn(self,X_train,y_train):
        self.theta = np.random.random(X_train.shape[1])
        max_iter = self.max_iter
        j_theta = []
        while max_iter > 0:
            gradient = np.dot(X_train.T, (self.sigmoid(X_train)-y_train))
            self.theta = self.theta - self.eta * gradient
            max_iter -= 1
            j_theta.append(self.getCost(X_train,y_train))
            
        if self.show_cost_graph == True:
            plt.plot(range(self.max_iter),j_theta)
            plt.xlabel("Number of iterations")
            plt.ylabel("Cost")
            plt.title("Number of iterations vs Cost (Logistic Regression)")
            plt.show()
            
    # Predict the classes for test dataset.    
    def predict(self,X_test):
        pred = np.zeros(X_test.shape[0])
        probabilities = self.sigmoid(X_test)
        y_0 = np.where(probabilities < 0.5)[0]
        y_1 = np.where(probabilities >= 0.5)[0]
        pred[y_0] = 0
        pred[y_1] = 1
        return pred

    # Calculate entries for confusion matrix.
    def get_confusionMatrix(self,y_actual,y_pred):
        true_one = len(np.where((y_actual == 1) & (y_pred == 1))[0])
        true_zero = len(np.where((y_actual == 0) & (y_pred == 0))[0])
        false_one = len(np.where((y_actual == 0)& (y_pred == 1))[0])
        false_zero = len(np.where((y_actual == 1)& (y_pred == 0))[0])
        cm = np.matrix([[true_zero,false_one],[false_zero,true_one]])
        return cm

    # Calculate classifiers measures.
    def get_measures(self,cm,show):
        acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
        sensitivity = (cm[0,0]) /(cm[0,0] + cm[1,0])
        specificity = (cm[1,1]) / (cm[1,1] + cm[0,1])
        print("Accuracy: %0.3f" % (acc*100),"%")
        print("Error: %0.3f" % ((1-acc)*100),"%")
        if show == True:
            print("Confusion Matrix : ")
            print(cm)
            print("Sensitivity: %0.3f" %sensitivity)
            print("Specificity: %0.3f" %specificity)
        return (1-acc)*100


errors_on_runs = []
for runs in range(10):
    logit = LogisticRegression(eta = 0.005, show_cost_graph = False)
    print("\nRun=",runs+1)
    print("Learning...")
    logit.learn(X_train, y_train)
    print("Predicting...")
    y_pred = logit.predict(X_test)
    cm = logit.get_confusionMatrix(y_test,y_pred)
    e = logit.get_measures(cm,False)
    errors_on_runs.append(e)
    print("Done")

print("Average Error:",np.mean(errors_on_runs),"+/-",np.std(errors_on_runs))
