import import_files

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataloader import load_susy_complete
from dataloader import splitdataset

#Load the data
train,test = load_susy_complete(2000,1000)
X_train, y_train = train
X_test, y_test = test


class LogisticRegression():
    """Logistic Regression with l1,l2 and elastic net regularizations.
       lambda1 = regularization for l1.
       lambda2 = regularization for l2.

       IMPORTANT:regularizer attribute takes on three values 'l1','l2' and 'elastic_net'    
    """

    def __init__(self,eta = 0.01, max_iter = 100,show_cost_graph = True, lambda_param1 = 0.1,lambda_param2 = 0.01,regularizer = "l1"):
        self.theta = None
        self.eta = eta
        self.max_iter = max_iter
        self.show_cost_graph = show_cost_graph
        self.lambda_param1 = lambda_param1
        self.lambda_param2 = lambda_param2
        self.regularizer = regularizer

    # Sigmoid function.
    def sigmoid(self,x):
        return 1/(1+ np.exp(- np.dot(self.theta,x.T)))

    #Derivative of l1 cost.
    def dl1(self,vec):
        """ Subgradient of l1 norm on a vector """
        grad = np.sign(vec)
        grad[abs(vec) < 1e-4] = 0.0
        return grad

    # Cost function for logistic regression.
    def getCost(self,X_train,y_train):
        cost = 0
        for i in range(X_train.shape[0]):
            cost += (y_train[i] * np.log(self.sigmoid(X_train[i]))) + ((1- y_train[i]) * np.log(1-self.sigmoid(X_train[i])))
        return (-cost/X_train.shape[0])

    # Signum function.
    def sgn(self,x):
        return x/np.abs(x)

    # Learning function for the regularized logistic regression.
    def learn(self,X_train,y_train):
        X_train = np.delete(X_train,-1,axis=1)
        self.theta = np.random.random(X_train.shape[1])#np.dot(np.dot(np.linalg.inv(np.dot(X_train.T,X_train)), X_train.T),y_train) #Just for testing the effect.
        max_iter = self.max_iter
        j_theta = []
        
        if self.regularizer == "l2":
            while max_iter > 0:
                gradient = np.dot(X_train.T, (self.sigmoid(X_train) - y_train))
                self.theta = self.theta * (1- (self.eta * self.lambda_param2)) - self.eta * gradient            
                max_iter -= 1
                j_theta.append(self.getCost(X_train,y_train))
            
        if self.regularizer == "l1":
            while max_iter > 0:
                gradient = np.dot(X_train.T, (self.sigmoid(X_train) - y_train))
                # Applying the soft thresholding condition, since we have the subgradient calculated we can directly apply it to the weight to update.
                idx = np.where(self.theta < self.lambda_param1)
                self.theta[idx] = self.theta[idx] - (self.eta * (gradient[idx] + self.lambda_param1 * np.apply_along_axis(self.dl1,0,self.theta)[idx]))
                idx = np.where(self.theta > -self.lambda_param1)
                self.theta[idx] = self.theta[idx] - (self.eta * (gradient[idx] - self.lambda_param1 * np.apply_along_axis(self.dl1,0,self.theta)[idx]))
                self.theta[np.where((self.theta > -self.lambda_param1) & (self.theta < self.lambda_param1))] = 0
                max_iter -= 1
                j_theta.append(self.getCost(X_train,y_train))

        if self.regularizer == "elastic_net":
            while max_iter > 0:
                # This version is the first approach. Notice the weights don't change when we send the weights to l1 regularizer gradient. Weights are updated at the
                #end of iteration which is really important or our method will be invalid.
                
                gradient = np.dot(X_train.T, (self.sigmoid(X_train)-y_train))
                self.theta = self.theta * (1- (self.eta * self.lambda_param2)) - self.eta * (gradient + self.lambda_param1 * np.apply_along_axis(self.dl1,0,self.theta))

                # This is the second approach to solving the equation. #Reference: http://web.stanford.edu/~hastie/Papers/elasticnet.pdf
                # UnComment next five lines if you wish to run the first approach. And comment the above two lines. Both give almost same results
                
                #gradient = np.dot(X_train.T, (self.sigmoid(X_train) - y_train))
                #self.theta = self.theta - self.eta * gradient
                #partial_penalty = (np.abs(self.theta) - self.lambda_param1/2)/(1+self.lambda_param2)
                #partial_penalty[np.where(partial_penalty<=0)] = 0
                #self.theta= partial_penalty * self.sgn(self.theta)
                max_iter -= 1
                j_theta.append(self.getCost(X_train,y_train))
            
        if self.show_cost_graph == True:
            plt.plot(range(self.max_iter),j_theta)
            plt.xlabel("Number of iterations")
            plt.ylabel("Cost")
            plt.title("Number of iterations vs Cost (Logistic Regression)")
            plt.show()

    # Prediction for the unknown examples.    
    def predict(self,X_test):
        X_test = np.delete(X_test,-1,axis=1)
        pred = np.zeros(X_test.shape[0])
        probabilities = self.sigmoid(X_test)
        y_1 = np.where(probabilities >= 0.5)[0]
        pred[y_1] = 1
        return pred

    # Calculate entries for confusion matrix.
    def get_confusionMatrix(self,y_actual,y_pred):
        true_one = len(np.where((y_actual == 1) & (y_pred == 1))[0])
        true_zero = len(np.where((y_actual == 0) & (y_pred == 0))[0])
        false_one = len(np.where((y_actual == 0)& (y_pred == 1))[0])
        false_zero = len(np.where((y_actual == 1)& (y_pred == 0))[0])
        cm = np.matrix([[true_zero,false_one],[false_zero,true_one]])
        print("Confusion Matrix :")
        print(cm)
        return cm

    # Calculate classifiers measures.
    def get_measures(self,cm):
        acc = (cm[0,0] + cm[1,1]) / np.sum(cm)
        sensitivity = (cm[0,0]) /(cm[0,0] + cm[1,0])
        specificity = (cm[1,1]) / (cm[1,1] + cm[0,1])
        print("Accuracy: %0.3f" % (acc*100),"%")
        print("Error: %0.3f" % ((1-acc)*100),"%")
        print("Sensitivity: %0.3f" %sensitivity)
        print("Specificity: %0.3f" %specificity)

logit = LogisticRegression(eta = 0.001, lambda_param1 = 0.01,lambda_param2 = 0.01)
logit.learn(X_train, y_train)
y_pred = logit.predict(X_test)
cm = logit.get_confusionMatrix(y_test,y_pred)
logit.get_measures(cm)
