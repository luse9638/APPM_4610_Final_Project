## Imports
##

import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt

## Error calculations
##

def calc_error(w, y, norm=2, type="rel"):
    """
    Calculate the error between vectors y and w as well as the norm of the error
    
    ### Parameters
        @w: vector of approximations
        @y: vector of actual values
        @norm: type of norm to use, default L2
        @type: type of error to use, default relative error

    ### Returns
        (error_vec, error_norm)
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

## Approximators
##

def RKm(m, f, a, b, alpha, N=0, h=0, debug=False):
    """
    Runs an mth order Runge-Kutta method
    Choose a value for h or N, but not both

    ### Parameters
        @m: Order of method
        @f: Differential equation y' = f(t, y)
        @a: left endpoint
        @b: right endpoint
        @alpha: y_0 = w_0 = alpha
        @N: number of intervals, (N + 1) total meshpoints including t = a
        @h: length of interval
        @debug: True to output debugging information, default False

    ### Returns
        (t_vec, w_vec)
    """

    # Debugging
    if debug:
        print("--------------------------------")
        print(f"Initializing order {m} Runge-Kutta...")
    
    # Check arguments were passed correctly
    if m <= 0:
        print("ERROR IN mStepExplicitAB(): NEED m > 0.")
        return
    if N == 0 and h == 0:
        print("ERROR IN RKm(): MUST SPECIFY VALUE FOR N or h.")
        return
    
    # Compute h and N, construct t_vec
    if N == 0:
        print("ERROR IN RKm(): NOT IMPLEMENTED CALCULATING h")
    elif h == 0:
        h = (b - a) / N
    t_vec = np.arange(a, b + h/10, h)

    # Initialize w_vec with initial data
    w_vec = np.zeros(N + 1)
    w_vec[0] = alpha

    # Compute b coefficiets based on order m
    # w_(j+1) = w_j + h_m[b_(m-1)f(t_j, w_j) + ... + b_(0)f(t_(i-m+1), w_(i-m+1))]
    k_vec = np.zeros(m)
    b_vec = np.zeros(m)
    h_m = 0
    if m == 2:
        h_m = h
        b_vec[0], b_vec[1] = 0, 1
        k_1j = lambda j: f(t_vec[j], w_vec[j])
        k_2j = lambda j: f(t_vec[j] + (h/2), w_vec[j] + (k_1j(j)/2))
        k_vec = lambda j: np.array([k_1j(j), k_2j(j)])
    elif m == 4:
        h_m = h/6
        b_vec[0], b_vec[1], b_vec[2], b_vec[3] = 1, 2, 2, 1
        k_1j = lambda j: f(t_vec[j], w_vec[j])
        k_2j = lambda j: f(t_vec[j] + (h/2), w_vec[j] + (h*k_1j(j) / 2))
        k_3j = lambda j: f(t_vec[j] + (h/2), w_vec[j] + (h*k_2j(j) / 2))
        k_4j = lambda j: f(t_vec[j+1], w_vec[j] + h*k_3j(j))
        k_vec = lambda j: np.array([k_1j(j), k_2j(j), k_3j(j), k_4j(j)])
    else:
        print("ERROR IN RKm(): THIS VALUE OF m NOT IMPLEMENTED")
        return
    
    # Debugging
    if debug:
        print(f"Order m = {m}")
        print(f"Step size h = {h}")
        print(f"Number of intervals N = {N}")
        print(f"Left endpoint a = {a}")
        print(f"Right endpoint b = {b}")
        print(f"Initial data: (t_0, w_0) = (a, alpha) = ({t_vec[0]}, {w_vec[0]})")
        print("\nRunning method...")
    
    # The method itself
    for j in range(0, N):
        w_vec[j+1] = w_vec[j] + h_m * (np.dot(k_vec(j), b_vec))

    return (t_vec, w_vec)
    
def mStepExplicitAB(m, f, a, b, alpha, order, N=0, h=0, debug=False):
    """ 
    Runs an m-step explicit Adams-Bashforth method
    Choose a value for h or N, but not both

    ### Parameters
        @m: Number of previous approximations to use in creation of next approximation
        @f: Differential equation y' = f(t, y)
        @a: left endpoint
        @b: right endpoint
        @alpha: y_0 = w_0 = alpha
        @order: order of RK method to use to create initial data
        @N: number of intervals, (N + 1) total meshpoints including t = a
        @h: length of interval
        @debug: True to output debugging information, default False

    ### Returns
        (t_vec, w_vec)
    """

    # Debugging
    if debug:
        print("----------------------------------------")
        print(f"Initializing {m}-step explicit Adams-Bashforth...")

    # Check arguments were passed correctly
    if m <= 0:
        print("ERROR IN mStepExplicitAB(): NEED m > 0.")
        return
    if N == 0 and h == 0:
        print("ERROR IN mStepExplicitAB(): MUST SPECIFY VALUE FOR N or h.")
        return
   
    # Compute h and N, construct t_vec
    if N == 0:
        print("ERROR IN mStepExplicitAB(): NOT IMPLEMENTED CALCULATING h")
    elif h == 0:
        h = (b - a) / N
    t_vec = np.arange(a, b + h/10, h)

    # Create initial data using RK and initialize w_vec
    # Endpoint b is set so length of init_vec is equivalent to m
    if debug:
        print("Constructing initial data using RK...")
    _, init_vec = RKm(order, f, a, a + h*(m-1), alpha, m - 1, debug=debug)
    w_vec = np.zeros(N + 1)
    for j in range(len(init_vec)):
        w_vec[j] = init_vec[j]

    # Compute b coefficients based on m
    # w_(j+1) = w_j + h_m[b_(m-1)f(t_j, w_j) + ... + b_(0)f(t_(i-m+1), w_(i-m+1))]
    b_vec = np.zeros(m)
    h_m = 0
    if m == 1:
        h_m = h
        b_vec[0] = 1
    elif m == 2:
        h_m = h/2
        b_vec[0], b_vec[1] = -1, 3
    elif m == 3:
        h_m = h/12
        b_vec[0], b_vec[1], b_vec[2] = 5, -16, 23
    elif m == 4:
        h_m = h/24
        b_vec[0], b_vec[1], b_vec[2], b_vec[3] = -9, 37, -59, 55
    elif m == 5:
        h_m = h/720
        b_vec[0], b_vec[1], b_vec[2], b_vec[3], b_vec[4] = 251, -1274, 2616, -2774, 1901
    else:
        print("ERROR IN mStepExplicitAB(): THIS VALUE OF m NOT IMPLEMENTED")
        return
    
    # Debugging
    if debug:
        print(f"Number of steps m = {m}")
        print(f"Step size h = {h}")
        print(f"Number of intervals N = {N}")
        print(f"Left endpoint a = {a}")
        print(f"Right endpoint b = {b}")
        print(f"Length of t_vec: {t_vec.shape}")
        print(f"Length of init_vec: {init_vec.shape}")
        print(f"Length of w_vec: {w_vec.shape}")
        print(f"Initial point: (t_0, w_0) = (a, alpha) = ({t_vec[0]}, {w_vec[0]})")
        print(f"Initial data generated from RK:")
        w_stat = [f"(t_{j}, w_{j}) = ({t_vec[j]}, {w_vec[j]})" for j in range(1, m)]
        for w in w_stat: print(w)
        print("\nRunning method...")

    # The method itself
    # Create the first m function evaluations
    f_eval_vec = np.array([f(t_vec[k], w_vec[k]) for k in range(0, m)])
    for j in range(m-1, N):
        # Compute next iteration
        w_vec[j+1] = w_vec[j] + h_m * (np.dot(f_eval_vec, b_vec))
        # Update function evaluations
        if j+1 < N:
            # Slide all elements to the left by 1
            f_eval_vec[:-1] = f_eval_vec[1:]
            # Replace last element with new evaluation
            f_eval_vec[-1] = f(t_vec[j+1], w_vec[j+1])
        if debug:
            print(f"j = {j}")
            t_stat = [f"(t_{k}, w_{k})" for k in range(j-m+1, j+1)]
            print(f"Function evaluations at: {t_stat}")
            print(f"(t_{j+1}, w_{j+1}) = ({t_vec[j+1]}, {w_vec[j+1]})")
            print("\n")

    return t_vec, w_vec