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

class CustomClassifier():
    """
    This is the alternate logistic regression classifier. The derivation of the likelihood function is the main.pdf.
    The first order derivative is used to update the weights on each iteration.
    """
    def __init__(self,max_iter,eta=1e-3):
        self.weights = None
        self.max_iter = max_iter
        self.eta = eta

    # The given pdf. 
    def pdf(self,wtx):
        return ((0.5) * (1+(wtx/np.sqrt(1+wtx**2))))

    # The first order derivative of the new logistic coast function.
    def first_order_derivative(self,wtx,y,x_ij):
        p = wtx/np.sqrt(1+wtx**2)
        dp_by_dw = x_ij/ ((1+wtx**2)**(1.5))
        return  ((y/(1+p))-((1-y)/(1-p))) * dp_by_dw

    def second_order_derivative(self,wtx,y,x_ij):
        p = wtx/np.sqrt(1+wtx**2)
        dp_by_dw = x_ij/ ((1+wtx**2)**(1.5))
        return ((y/(1+p)**2) + (1-y)/(1-p)**2) * (dp_by_dw)**2 + ((y/(1+p))-((1-y)/(1-p))) * (-2*x_ij/(1+wtx**2)**2)

    # The logistic cost function.
    def getCost(self,X_train,y_train):
        cost = 0
        for i in range(X_train.shape[0]):
            prob_density = self.pdf(np.dot(self.weights.T,X_train[i]))
            cost += (np.log((prob_density))*(y_train[i])) + (np.log(1-prob_density)*(1-y_train[i]))
        return cost/X_train.shape[0]

    # The learning function for the alternate logistic regression.        
    def learn(self,X_train,y_train):
        self.weights = np.random.random(X_train.shape[1])
        weights = np.random.random(X_train.shape[1]) # These are temporary weights.
        max_iter = self.max_iter
        j_theta = []
        while max_iter > 0:
            for j in range(X_train.shape[1]):
                first_order_der = 0
                second_order_der = 0
                for i in range(X_train.shape[0]):
                    wtx = np.dot(self.weights.T,X_train[i])
                    # sum over all the training examples.
                    first_order_der += self.first_order_derivative(wtx,y_train[i],X_train[i,j])
                    second_order_der += self.second_order_derivative(wtx,y_train[i],X_train[i,j])
                weights[j] = self.weights[j] - (self.eta * (first_order_der)/(second_order_der)) #We need to keep the weights same for a feature so this is temp placeholder.
            self.weights = weights #We change all the weights here. 
            j_theta.append(self.getCost(X_train,y_train))
            max_iter -= 1
        plt.plot(range(self.max_iter),j_theta)
        plt.xlabel("Number of iterations")
        plt.ylabel("Cost")
        plt.title("Number of iterations vs Cost (Custom Logistic Regression)")
        plt.show()

    # Predict the unknown test samples using the above learned function.
    def predict(self,X_test):
        pred = np.zeros(X_test.shape[0])
        y_test_pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            y_test_pred[i] = self.pdf(np.dot(self.weights.T,X_test[i]))
        idx = np.where(y_test_pred > 0.5)
        pred[idx] = 1
        return pred
    
    def get_confusionMatrix(self,y_actual,y_pred):
        true_one = len(np.where((y_actual == 1) & (y_pred == 1))[0])
        true_zero = len(np.where((y_actual == 0) & (y_pred == 0))[0])
        false_one = len(np.where((y_actual == 0)& (y_pred == 1))[0])
        false_zero = len(np.where((y_actual == 1)& (y_pred == 0))[0])
        cm = np.matrix([[true_zero,false_one],[false_zero,true_one]])
        print(cm)
        return cm
    
    # Calculate classifiers measures.
    def get_measures(self,cm):
        acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
        sensitivity = (cm[0,0]) /(cm[0,0] + cm[1,0])
        specificity = (cm[1,1]) / (cm[1,1] + cm[0,1])
        print("Accuracy: %0.3f" % (acc*100),"%")
        print("Sensitivity: %0.3f" %sensitivity)
        print("Specificity: %0.3f" %specificity)
    
customClassifier = CustomClassifier(max_iter = 200)
customClassifier.learn(X_train,y_train)
y_pred = customClassifier.predict(X_test)
cm = customClassifier.get_confusionMatrix(y_test,y_pred)
customClassifier.get_measures(cm)
