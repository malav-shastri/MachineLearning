#If code throws error replace matmul function with dot function. My version supports matmul and not dot.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

#to calculate hypothesis
def sigmoid(features_train,theta):
    z = 1/(1 + np.exp(-hypothesis(features_train,theta)))
    return z

#To calculate x*theta
def hypothesis(features_train, theta):
    temp = np.matmul(theta,features_train.T)
    return temp

#To predict the values from theta parameters and hypothesis function
def predict(x, y, theta):
    predicted = sigmoid(x,theta)
    predicted1 = predicted
    for i in range(len(predicted[0])):
        if(predicted[0][i]>=0.5):
            predicted[0][i] = 1
        else:
            predicted[0][i] = 0
    print("Predicted Values")
    print(predicted.astype(int))
    return predicted1

#Maximum Likelihood estimation uses gradient ascent rather then gradient ascent function
def gradient_ascent(X,theta,y):
    t = np.matmul(X.T, y - (sigmoid(X,theta).T))
    return t

#Unlike gradient decent in gradient ascent you add older weights in gradient*learning rate
def update_weight_mle(weight, learning_rate, gradient):
    pheta = np.multiply(gradient,learning_rate)
    uwm = np.add(weight,pheta.T)
    return uwm

#To plot the scatter plot for testing data
def plot_output(X,y):
    X_new = X.iloc[:, 1:]
    #print(X_new)
    temp = X_new[y == 0]
    temp2 = X_new[y ==1]
    plt.scatter(temp.iloc[:, 0], temp.iloc[:, 1], color='r', label='0 = non-virginica')
    plt.scatter(temp2.iloc[:, 0], temp2.iloc[:, 1], color='b', label='1 = virginica')
    plt.legend()
    plt.xlabel('sepal length', fontsize='13')
    plt.ylabel('sepal width', fontsize='13')
    plt.title('Logistic Regression output for the test data-set', fontsize='13')
    plt.show()

#To find the likelihood, likelihood should increase with interation in MLE
def log_likelihood(x, y, weights):
    ll = np.sum(np.matmul(y,hypothesis(x,weights)) - np.log(1 + np.exp(hypothesis(x,weights))))
    return ll 

#read and shuffle the data
iris = pd.read_csv(r'D:\Stevens\Fall 19\ML\Assignment 1\IRIS.csv')
iris = shuffle(iris)

iris.info()
mapping = {
    'Iris-virginica' : 1,
    'Iris-versicolor': 0,
    'Iris-setosa'    : 0
}
rows, col = iris.shape
bias = np.ones(len(iris))
X = iris.loc[:,['sepal_length','sepal_width']]
X.insert(0,'bias',bias)
lables = iris.species.replace(mapping).values.reshape(rows,1)

feature_train, feature_test = X[0:99], X[100:]
lables_train, lables_test = lables[0:99], lables[100:]

theta = np.zeros(3,dtype='int')
theta = theta.reshape(1,3)
learning_rate = 0.001
iters = 1000000
likelihood = np.zeros(iters)
#To update theta gradient and likelihood
for i in range(iters):
    gradient = gradient_ascent(feature_test,theta,lables_test)
    theta = update_weight_mle(theta, learning_rate, gradient)
    likelihood[i] = log_likelihood(feature_train, lables_train, theta)
print(likelihood)
print(theta)
#To plot likelihood vs iteration...with iteration likelihood should increase
predicted1 = predict(feature_train, lables_train, theta)
plot_output(feature_test, lables_test)
plt.plot(np.arange(iters),likelihood)
plt.xlabel('Number of iterations',color='red')
plt.ylabel('Likelihood',color='red')
plt.title('Likelihood vs iterations',color='green')
plt.show()
