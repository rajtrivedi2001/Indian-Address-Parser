from numpy import *
import numpy

def regularize(X) :
    """Function is used to normalize the input matrix based column wise
    Returns:
        [reg_X,mu,std]
        reg_X - normalized values
        mu - mean values
        std - standard deviation values
    """
    mu = numpy.mean(X,0)
    std = numpy.std(X,0)
    reg_X = (X - mu)/std
    return (reg_X,mu,std)
