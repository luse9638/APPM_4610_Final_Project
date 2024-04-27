# Imports ######################################################################
################################################################################

import numpy as np
import scipy as sp

# Subroutines ##################################################################
################################################################################

def calc_BDF_order():
    '''
    
    '''

def calc_stepsize():
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
    
    # Base case.
    if start == end:
        return y_nodes[end] 
    
    # Recursive case.
    return (newton_div_diff(t_nodes, y_nodes, start+1, end) - newton_div_diff(t_nodes, y_nodes, start, end-1))\
        / (t_nodes[end] - t_nodes[start])

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

        # Current time step approximation loop.
        # Continue looping until our computed approximation meets certain
        # condtions.
        while True: # TODO: fix this condition
            # Determine order of BDF method to use.
            k_j = calc_BDF_order() # TODO: write this function

            # Determine a stepsize to use.
            h_j = calc_stepsize() # TODO: write this function.

            # Get the previous (k+1) approximations to use as interpolant nodes
            # in the predictor polynomial.
            t_pred_nodes = [t_vec[j-jj] for jj in range(k_j+1)] # jj in [0, k_j]
            w_pred_nodes = [w_vec[j-jj] for jj in range(k_j+1)]

            # Form the predictor polynomials

