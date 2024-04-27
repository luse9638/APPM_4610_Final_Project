# Imports ######################################################################
################################################################################

import numpy as np
import scipy as sp
import math

# Subroutines ##################################################################
################################################################################

def calc_BDF_order(): 
    '''
    
    '''

def calc_stepsize(): # Should have parameter for whether its a newton stepsize calculation
    '''
    '''

def newton_div_diff(t_nodes, y_nodes, start, end):
    '''
    Find the divided difference [y_end, y_{end-1}, ..., y_start]
    
    @t_nodes: vector of t values
    @y_nodes: vector of y values
    @start: start index
    @end: end index
    '''

    # Predicates.
    assert len(t_nodes) == len(y_nodes)
    assert start <= end
    
    # Base case.
    if start == end:
        return y_nodes[end] 
    
    # Recursive case.
    return (newton_div_diff(t_nodes, y_nodes, start+1, end) - newton_div_diff(t_nodes, y_nodes, start, end-1))\
        / (t_nodes[end] - t_nodes[start])

def eval_interp_poly(t_nodes, y_nodes, t_eval):
    '''
    Form an interpolating polynomial and evaluate it at a point
    
    ### Parameters
    @t_nodes: vector of t values
    @y_nodes: vector of y values
    @t_eval: what to evaluate polynomial at

    ### Returns
    double
    '''

    # Predicates.
    assert len(t_nodes) == len(y_nodes)

    n = len(t_nodes)
    eval = 0
    for i in range(0, n):
        term = newton_div_diff(t_nodes, y_nodes, 0, i)
        for j in range(0, i):
            term *= t_eval - t_nodes[j]
        eval += term

    return eval
    
def psi(h_vec, t_vec, i, j):
    '''
    Compute psi_i(j+1)
    '''
    assert i >= 1
    
    h_val = sum(h_vec[jj] for jj in range(j+2-i, j+2))
    t_val = t_vec[j+1] - t_vec[j+1-i]

    # Should be the same
    assert h_val == t_val
    return t_val

def alpha(h_vec, t_vec, i, j):
    '''
    Compute alpha_i(j+1)
    '''
    assert i >= 1

    return h_vec[j+1] / psi(h_vec, t_vec, i, j)

def beta(h_vec, t_vec, i, j):
    '''
    Compute beta_i(j+1)
    '''
    if i == 1:
        return 1
    assert i > 1
    numer = 1
    denom = 1
    for ii in range(1, i): # ii in [1, i-1]
        numer *= psi(h_vec, t_vec, ii, j)
        denom *= psi(h_vec, t_vec, ii, j-1)
    return numer/denom
    
def phi(h_vec, t_vec, y_vec, i, j):
    '''
    Compute phi_i(j)
    '''
    if i == 1:
        return y_vec[j]
    assert i > 1
    prod = 1
    for ii in range(1, i): # ii in [1, i-1]
        prod *= psi(h_vec, t_vec, ii, j-1)
    return prod * newton_div_diff(t_vec, y_vec, j-i+1, j)

def phi_star(h_vec, t_vec, y_vec, i, j):
    '''
    Compute phi^{*}_i(j)
    '''
    assert i >= 1
    return beta(h_vec, t_vec, i, j) * phi(h_vec, t_vec, y_vec, i, j)

def sigma(h_vec, t_vec, i, j):
    '''
    Compute sigma_i(j+1)
    '''
    if i == 1:
        return 1
    assert i > 1
    numer = (h_vec[j+1] ** i) * math.factorial(i - 1)
    denom = 1
    for ii in range(1, i+1): # ii in [1, i]
        denom *= psi(h_vec, t_vec, ii, j)
    return numer / denom

def gamma(h_vec, t_vec, i, j):
    '''
    Compute gamma_i(j+1)
    '''
    if i == 0:
        return 0
    assert i > 1
    return gamma(h_vec, t_vec, i-1, j) + (alpha(h_vec, t_vec, i-1, j) / h_vec[j+1])

def alpha_s(k):
    '''
    Compute alpha_s
    '''
    return -sum(1/jj for jj in range(1, k+1)) # jj in [1, k]

def alpha_0(h_vec, t_vec, j, k):
    '''
    Compute alpha^{0}(j+1)
    '''
    return -sum(alpha(h_vec, t_vec, ii, j) for ii in range(1, k+1)) # ii in [1, k]

# DASSL ########################################################################
################################################################################

def scalar_dassl(f, t_0, t_f, y_0, dy_0, h_0, newt_tol, debug=True):
    '''
    @f: f(t, y, y') = 0
    @t_0: left endpoint
    @t_f: right endpoint
    @y_0: y(t_0) = y_0
    @dy_0: y'(t_0) = dy_0
    @h_0: initial stepsize
    @newt_tol: successive approximation tolerance to use for terminating Newton's method
    '''

    if debug:
        print(f"Running scalar_dassl on [{t_0}, {t_f}] with y({t_0}) = {y_0}, y'({t_0}) = {dy_0}, and initial stepsize h = {h_0}")
    
    # Initialize vectors.
    t_vec = [t_0]
    w_vec = [y_0]
    dw_vec = [dy_0]
    h_vec = []
    k_vec = [] # TODO: how to select first BDF order?

    # Conditions for terminating the time stepping.
    interval_len = t_f - t_0
    interval_cov = 0
    covered_interval = False
    j = 0
    
    # Time stepping loop.
    # Each iteration (j) uses t_j, k_{j-1}, h_{j-1}, w_j, and dw_j to calculate
    # h_j and k_j in order to compute t_{j+1}, w_{j+1}, and dw_{j+1}.
    while not covered_interval:
        # Get values from the previous time step, t_{j}.
        t_j = t_vec[-1]
        w_j = w_vec[-1]
        dw_j = dw_vec[-1]
        # If we're creating w_1 (j = 0), we don't have previous values for
        # k_{j-1} and h_{j-1}.
        if j == 0:
            h_jm1 = None 
            k_jm1 = None
        else:
            h_jm1 = h_vec[-1] # t_{j+1} = t_j + h_j.
            k_jm1 = k_vec[-1]
        
        # Vectors of stepsizes, approximations, and BDF orders that have been
        # trialed for the current time step (j).
        w_jp1_trial_vec = []
        dw_jp1_trial_vec = []
        h_j_trial_vec = []
        k_j_trial_vec = []

        # Current time step (j) approximation loop.
        # Continue looping until our computed approximation meets certain
        # condtions.
        while True: # TODO: fix this condition
            # Determine order of BDF method to use.
            k_j = calc_BDF_order() # TODO: write this function

            # Determine a stepsize to use.
            h_j = calc_stepsize() # TODO: write this function.

            # Get the previous (k+1) approximations to use as interpolant nodes
            # in the predictor polynomial.
            # TODO: maybe this should only be k nodes when creating w_1?
            t_pred_nodes = [t_vec[j-jj] for jj in range(k_j+1)] # jj in [0, k_j]
            w_pred_nodes = [w_vec[j-jj] for jj in range(k_j+1)]
            dw_pred_nodes = [dw_vec[j-jj] for jj in range(k_j+1)]

            # Vector of stepsizes trialed for Newton's method at time step (j).
            h_j_newt_vec = []

            # Newton's method loop, we continue retrying with different initial
            # guesses (from trialing different stepsizes) until we are within
            # the specified tolerance error.
            newt_iter_count = 0
            while newt_iter_count < 10: # Terminate the whole algorithm if
                                        # Newton's refuses to converge.
                # Calculate h_newt_j and use to create initial approximations
                h_j_newt = calc_stepsize() # TODO: write this function
                t_jp1 = t_j + h_j_newt
                w0_jp1 = eval_interp_poly(t_pred_nodes, w_pred_nodes, t_jp1)
                dw0_jp1 = eval_interp_poly(t_pred_nodes, dw_pred_nodes, t_jp1) # TODO: is there a better way to do this?


    