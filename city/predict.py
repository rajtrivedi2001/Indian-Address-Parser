from numpy import *
from sigmoid import sigmoid

def predict(X,theta) :
    """Function returns the prediction for a record(X),
    for the given weights(theta)
    """
    
    m = X.shape[0]
    X = concatenate((mat(zeros((m,1))),X),1)
    z = X*theta
    h = sigmoid(z)
    return h[0,0]
