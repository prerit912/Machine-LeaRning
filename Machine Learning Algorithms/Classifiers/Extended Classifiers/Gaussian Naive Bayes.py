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
Implementation of Naive Bayes Classifier.

Attributes: class_prior - Prior probabilities.
            class_count - unique counts of classes.
            mu_ml = Maximum Likelihood of mean parameter. shape = (#classes,#features)
            sigma_sq_ml = Maximum Likelihood of variance parameter. shape = (#classes,#features)
            classes = unique classes.
            remove_ones = boolean value to include features of one.
"""

class NaiveBayes():
    """ Gaussian naive Bayes """

    # Constructor
    def __init__( self, remove_ones = True ):
        self.class_prior = None
        self.class_count = None
        self.mu_ml = None
        self.sigma_sq_ml = None
        self.classes = None
        self.remove_ones = remove_ones

    # PDF for Gaussian Distribution.
    def Gaussian_pdf(self,x,mu,sigma_sq):
        return (1/np.sqrt(2*np.pi*sigma_sq)) * (np.exp(-np.square((x - mu))/ (2*sigma_sq)))

    # Learn Naive bayes classifier.
    def learn(self,X_train,y_train,remove_ones = True):
        if self.remove_ones == True:
            X_train = np.delete(X_train,-1,axis = 1)
        # Set initial parameter values to useful to predictions.
        unique_classes,unique_counts = np.unique(y_train,return_counts = True)
        self.classes = unique_classes
        unique_class_count = len(unique_classes) 
        self.class_prior = unique_counts/X_train.shape[0] 
        self.class_count = unique_counts 
        self.mu_ml = np.zeros((unique_class_count,X_train.shape[1]))
        self.sigma_sq_ml = np.zeros((unique_class_count,X_train.shape[1]))

        nth_class = -1
        for c in np.unique(y_train):
            nth_class += 1
            class_indices = np.where(y_train == c)
            X_class_train = X_train[class_indices,][0]

            for j in range(X_train.shape[1]):    
                self.mu_ml[nth_class,j] = np.mean(X_class_train[:,j])
                self.sigma_sq_ml[nth_class,j] = np.mean(np.square(X_class_train[:,j] - self.mu_ml[nth_class,j]))

    # Predict from learned parameters.
    def predict(self,X_test):
        predicted_classes = np.empty(X_test.shape[0])
        if self.remove_ones == True:
            X_test = np.delete(X_test,-1,axis = 1)
        for i in range(X_test.shape[0]):
            likelihood = np.zeros((len(self.classes),X_test.shape[1]))
            for j in range(X_test.shape[1]):
                nth_class = 0
                while nth_class < len(self.classes):
                    likelihood[nth_class,j] = self.Gaussian_pdf(X_test[i,j],self.mu_ml[nth_class,j],self.sigma_sq_ml[nth_class,j])
                    nth_class += 1
            unNormalized_posterior = np.prod(likelihood,axis = 1) * self.class_prior
            Normalized_posterior = unNormalized_posterior/np.sum(unNormalized_posterior)
            predicted_classes[i] = self.classes[np.where(Normalized_posterior == np.max(Normalized_posterior))]
        return predicted_classes

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
        if show == True:
            print("Confusion Matrix : ")
            print(cm)
            print("Accuracy: %0.3f" % (acc*100),"%")
            print("Error: %0.3f" % ((1-acc)*100),"%")
            print("Sensitivity: %0.3f" %sensitivity)
            print("Specificity: %0.3f" %specificity)
        return (1-acc)*100
    
        
                
errors_on_runs = []
for runs in range(10):
    gaussianNB = NaiveBayes()
    print("Run=",runs+1)
    print("Learning...")
    gaussianNB.learn(X_train,y_train)
    print("Predicting...")
    y_pred = gaussianNB.predict(X_test)
    cm = gaussianNB.get_confusionMatrix(y_test,y_pred)
    e = gaussianNB.get_measures(cm,False)
    errors_on_runs.append(e)
    print("Done")

print("Average Error:",np.mean(errors_on_runs),"+/-",np.std(errors_on_runs))
