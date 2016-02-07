from numpy import *
from sigmoid import sigmoid

def costFunction(theta,X,y) :
    """Function returns the logistic cost funtion"""
        
    m = X.shape[0]

    #adding a column  of ones as the first column of X
    X = concatenate((mat(zeros((m,1))),X),1)

    z = X*theta
    h = sigmoid(z)
    J = -sum( multiply(y,log(h)) + multiply((1-y),log(1-h)) )/m ;
    
    return J
