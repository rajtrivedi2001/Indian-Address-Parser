from math import *
from numpy import *
def sigmoid(z) :
    """Function returns the sigmoid of an n-d matrix"""
    return 1./(1+exp(-z))
