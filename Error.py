## Imports
##

import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

## Subroutines
##

def calcError(w, y, norm=2, type="rel"):
    """
    Calculate the error between vectors y and w as well as the norm of the error
    
    ### Parameters
        @w: vector of approximations
        @y: vector of actual values
        @norm: type of norm to use, default L2
        @type: type of error to use, default relative error

    ### Returns
    """

    # Make sure error and actual vector are the same length
    if len(w) != len(y):
        print("ERROR IN calcError(): VECTORS ARE NOT SAME LENGTH")
        return
    
    error_vec = np.zeros(len(w))
    for j in range(0, len(w)):
        if type == "rel":
            error_vec[j] = abs(w[j] - y[j]) / abs(y[j])
        elif type == "abs":
            error_vec[j] = abs(w[j] - y[j])
        
    err_norm = np.linalg.norm(error_vec, ord=norm)

    return error_vec, err_norm