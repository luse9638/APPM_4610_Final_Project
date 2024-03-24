## Imports
##

import numpy as np
import scipy as sp
import math

## Subroutines
##

def mStepExplicitAB(m, f, a, b, alpha_vec, N=0, h=0, debug=False):
    """ 
    Runs an m-step explicit Adams-Bashforth method
    Choose a value for h or N, but not both

    ### Parameters
        @m: Number of previous approximations to use in creation of next approximation
        @f: Differential equation y' = f(t, y)
        @a: left endpoint
        @b: right endpoint
        @alpha_vec: y_0 = w_0 = alphaVec[0], ..., w_(m-1) = alphaVec[m-1]
        @N: number of intervals, (N + 1) total meshpoints including t = a
        @h: length of interval
        @debug: True to output debugging information, default False

    ### Returns

    ### Raises
    """

    # Check arguments were passed correctly
    if m <= 0:
        print("ERROR IN mStepExplicitAB(): NEED m > 0.")
        return
    if len(alpha_vec) != m:
        print("ERROR IN mStepExplicitAB(): AMOUNT OF INITIAL DATA DOES NOT MATCH VALUE OF m")
        return
    if N == 0 and h == 0:
        print("ERROR IN mStepExplicitAB(): MUST SPECIFY VALUE FOR N or h.")
        return
   
    # Compute h, N, and t_vec
    if N == 0:
        print("ERROR IN mStepExplicitAB(): NOT IMPLEMENTED CALCULATING h")
    elif h == 0:
        h = (b - a) / N
    t_vec = np.arange(a, b + h/10, h)

    # Compute coefficients based on m
    # w_(j+1) = w_j + h_m[b_(m-1)f(t_j, w_j) + ... + b_(0)f(t_(i-m+1), w_(i-m+1))]
    # https://en.wikipedia.org/wiki/Linear_multistep_method#Adams%E2%80%93Bashforth_methods
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
    
    # Create w_vec
    w_vec = np.zeros(N + 1)
    for j in range(len(alpha_vec)):
        w_vec[j] = alpha_vec[j]
    
    # Debugging
    if debug:
        print("-------------------")
        print(f"m = {m}")
        print(f"h = {h}, N = {N}")
        print(f"a = {a}, b = {b}")
        print(f"(t_0, y({a})) = (t_0, w_0) = ({t_vec[0]}, {w_vec[0]})")
        w_stat = [f"(t_{j}, w_{j}) = ({t_vec[j]}, {w_vec[j]})" for j in range(1, m)]
        for w in w_stat: print(w)
        print("\nRunning method...")

    # The method itself
    f_eval_vec = np.array([f(t_vec[k], w_vec[k]) for k in range(0, N + 1)])
    for j in range(m - 1, N):
        w_vec[j+1] = w_vec[j] + h_m * (np.dot(f_eval_vec[j-m+1:j+1], b_vec))
        if debug:
            print(f"j = {j}")
            t_stat = [f"(t_{k}, w_{k})" for k in range(j-m+1, j+1)]
            print(f"Function evaluations at: {t_stat}")
            print(f"(t_{j+1}, w_{j+1}) = ({t_vec[j+1]}, {w_vec[j+1]})")
            print("\n")

def driver():
    f = lambda t, y: 2 * t
    mStepExplicitAB(4, f, 0, 4, [0, 0.0064, 0.0256, 0.0576], 50, debug=True)

driver()