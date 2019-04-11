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
Implementing Neural Network with one hidden layer.
    output_layer_weights = weights on the output layer.
    hidden_layer_weights = weights on the hidden layer.
    eta = learning rate
    max_iter = maximum iterations.
"""
class NeuralNets():
    def __init__(self,eta = 0.01,show_cost_graph = True):
        self.output_layer_weights = None
        self.hidden_layer_weights = None
        self.eta = eta
        self.max_iter = 100
        self.show_cost_graph = show_cost_graph
        
    def sigmoid(self,h):
        return 1/(1+ np.exp(-h))

    def learn(self,X_train,y_train):
        self.hidden_layer_neurons = int(X_train.shape[1]) + 1
        total_iterations = self.max_iter #saving the total iterations.
        
        # Weights are randomly initialized with uniform distribution.
        r = 0.5
        self.output_layer_weights = np.random.uniform(-r,r,self.hidden_layer_neurons)
        self.hidden_layer_weights = np.random.uniform(-r,r,(self.hidden_layer_neurons,X_train.shape[1]))
        
        hidden_layer_activation = np.zeros(self.hidden_layer_neurons)
        pred = np.zeros(X_train.shape[0])
        err_hidden = np.zeros(self.hidden_layer_neurons)
        err = [] # list of errors on each iterations, used for ploting.
        while self.max_iter >0:
            e = 0 # error in each iteration.
            for i in range(X_train.shape[0]):

                # Forward Phase
                hidden_layer_activation[0] = 1 #bias unit for output
                h = np.dot(X_train[i], self.hidden_layer_weights[0].T)
                hidden_layer_activation[0] = self.sigmoid(h)

                # Calculate hidden layer activations.
                for j in range(1,self.hidden_layer_neurons):
                    h = np.dot(X_train[i], self.hidden_layer_weights[j].T)
                    hidden_layer_activation[j] = self.sigmoid(h)

                # Find outputs.
                h = np.dot(hidden_layer_activation, self.output_layer_weights.T)
                pred[i] = self.sigmoid(h)
                
                # Backward Phase
                err_output = (-y_train[i] * (1-pred[i])) + ((1-y_train[i]) * (pred[i]))
                e += err_output

                # Calculate the hidden layer weights.
                for j in range(self.hidden_layer_neurons):
                    err_hidden[j] = hidden_layer_activation[j] * (1 - hidden_layer_activation[j]) * self.output_layer_weights[j] * err_output
                    self.output_layer_weights[j] = self.output_layer_weights[j] - self.eta * err_output * hidden_layer_activation[j]
                self.hidden_layer_weights = self.hidden_layer_weights - self.eta * np.dot(err_hidden.reshape(self.hidden_layer_neurons,1), X_train[i].reshape(X_train.shape[1],1).T)
            err.append(e/X_train.shape[0])
            self.max_iter -= 1
            
            # Shuffle the training data, so that our network doesn't learn in same order on every iteration.
            shuffler = np.arange(X_train.shape[0])
            np.random.shuffle(shuffler)
            X_train = X_train[shuffler,:]
            y_train = y_train[shuffler]

        if self.show_cost_graph == True:
            plt.plot(list(range(total_iterations)), err)
            plt.xlabel("Iterations")
            plt.ylabel("Cost")
            plt.title("Iterations vs Cost (ANN)")
            plt.show()

    def predict(self,X_test):
        hidden_layer_activation = np.zeros(self.hidden_layer_neurons)
        pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            for j in range(self.hidden_layer_neurons):
                h = np.dot(X_test[i], self.hidden_layer_weights[j].T)
                hidden_layer_activation[j] = self.sigmoid(h)
            
            h = np.dot(hidden_layer_activation, self.output_layer_weights.T)
            if (self.sigmoid(h) >= 0.5):
                pred[i] = 1
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
            print("Accuracy: %0.3f" % (acc*100),"%")
            print("Error: %0.3f" % ((1-acc)*100),"%")
            print("Sensitivity: %0.3f" %sensitivity)
            print("Specificity: %0.3f" %specificity)
        return (1-acc)*100
    
errors_on_runs = []
for runs in range(10):
    neuralNet = NeuralNets(eta = 0.01,show_cost_graph = False) #0.01
    print("\nRun=",runs+1)
    print("Learning...")
    neuralNet.learn(X_train,y_train)
    print("Predicting...")
    y_pred = neuralNet.predict(X_test)
    cm = neuralNet.get_confusionMatrix(y_test,y_pred)
    e = neuralNet.get_measures(cm,False)
    errors_on_runs.append(e)
    print("Done")

print("Average Error:",np.mean(errors_on_runs),"+/-",np.std(errors_on_runs))
