from numpy import *
from sigmoid import sigmoid

def grad(theta,X,y) :
    """Function returns the gradient of weights(theta)"""    
        
    #adding a column of ones as the first column of X
    m = X.shape[0] 
    X = concatenate((mat(zeros((m,1))),X),1)
    h = sigmoid(X*theta)
    grad = ((X.transpose())*(h-y))/m ;
    return grad
