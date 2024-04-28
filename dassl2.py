# Imports ######################################################################
################################################################################

import numpy as np
import scipy as sp
import math

# Subroutines ##################################################################
################################################################################

def calc_ord_stepsize(h_vec, t_vec, y_vec, h_prev, j, k, num_fail):
    '''
    '''
    # Estimate leading term of Taylor series expansions for orders (k-2), (k-1),
    # and k, and use these to calculate k_new.
    if k-2 >= 1:
        tay_km2 = abs((k-1) * sigma(h_vec, t_vec, k-1, j) * phi(h_vec, t_vec, y_vec, k, j+1))
        tay_km1 = abs(k * sigma(h_vec, t_vec, k, j) * phi(h_vec, t_vec, y_vec, k+1, j+1))
        tay_k = abs((k+1) * sigma(h_vec, t_vec, k+1, j) * phi(h_vec, t_vec, y_vec, k+2, j+1))
        if tay_km2 <= tay_km1 and tay_km1 <= tay_k:
            k_new = k-1
        elif tay_k <= tay_km1 and tay_km1 <= tay_km2:
            k_new = k+1
        else:
            k_new = k
    elif k-1 >= 1:
        tay_km1 = abs(k * sigma(h_vec, t_vec, k, j) * phi(h_vec, t_vec, y_vec, k+1, j+1))
        tay_k = abs((k+1) * sigma(h_vec, t_vec, k+1, j) * phi(h_vec, t_vec, y_vec, k+2, j+1))
        if tay_km1 <= tay_k:
            k_new = k-1
        elif tay_k <= tay_km1:
            k_new = k+1
        else:
            k_new = k
    elif k >= 1:
        k_new = k

    # Determine new step size, h_new
    est = (1/(k_new+1)) * abs(k_new * sigma(h_vec, t_vec, k_new, j) * phi(h_vec, t_vec, y_vec, k_new+1, j))
    r = (2*est) ** (-1/(k_new+1))
    if num_fail == 0 and r >= 2: # Last step was successful and we can at least double the step size
        h_new = 2*h_prev
    elif num_fail == 0 and r >= 1: # Last step was successful but we can't double the step size
        h_new = h_prev
    elif num_fail == 0 and r < 1: # Last step was successful but we want to decrease the step size
        h_new = 0.9*h_prev
    elif num_fail == 1: # Last step failed so we decrease is called for
        h_new = 0.9*r*h_prev
    elif num_fail > 1: # Last step has failed multiple times
        h_new = 0.25*h_prev
    else:
        print("ERROR IN calc_ord_stepsize(): UNSURE HOW TO ADJUST STEPSIZE")
        return
    
    return k_new, h_new

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

    if i == 1:
        h_val = h_vec[-1]
        t_val = t_vec[-1] - t_vec[-2]
    else:
        h_val = sum(h_vec[-i:])
        t_val = t_vec[-1] - t_vec[-i]
    
    print(f"        psi_{i}({j}+1) = {h_val} (h_val) = {t_val} (t_val)")

    # Should be the same
    #assert h_val == t_val
    return t_val

def alpha(h_vec, t_vec, i, j):
    '''
    Compute alpha_i(j+1)
    '''
    assert i >= 1

    val = h_vec[j] / psi(h_vec, t_vec, i, j)
    print(f"        alpha_{i}({j}+1) = {val}")

    return val

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
    val = numer/denom
    print(f"        beta_{i}({j}+1) = {val}")
    return val
    
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
    val = prod * newton_div_diff(t_vec, y_vec, j-i+1, j)
    print(f"        beta_{i}({j}) = {val}")
    return val

def phi_star(h_vec, t_vec, y_vec, i, j):
    '''
    Compute phi^{*}_i(j)
    '''
    assert i >= 1
    val = beta(h_vec, t_vec, i, j) * phi(h_vec, t_vec, y_vec, i, j)
    print(f"        phi_star_{i}({j}) = {val}")
    return val

def sigma(h_vec, t_vec, i, j):
    '''
    Compute sigma_i(j+1)
    '''
    if i == 1:
        return 1
    assert i > 1
    numer = (h_vec[j-1] ** i) * math.factorial(i - 1)
    denom = 1
    for ii in range(1, i+1): # ii in [1, i]
        denom *= psi(h_vec, t_vec, ii, j)
    val = numer/denom
    print(f"        sigma_{i}({j}+1) = {val}")
    return numer / denom

def gamma(h_vec, t_vec, i, j):
    '''
    Compute gamma_i(j+1)
    '''
    if i == 0:
        return 0
    assert i > 1
    val = gamma(h_vec, t_vec, i-1, j) + (alpha(h_vec, t_vec, i-1, j) / h_vec[j])
    print(f"        gamma_{i}({j}+1) = {val}")
    return val

def alpha_s(k):
    '''
    Compute alpha_s
    '''

    val = -sum(1/jj for jj in range(1, k+2)) # jj in [1, k+1]
    print(f"        alpha_s (k = {k}+1) = {val}")
    return val

def alpha_0(h_vec, t_vec, j, k):
    '''
    Compute alpha^{0}(j+1)
    '''
    val = -sum(alpha(h_vec, t_vec, ii, j) for ii in range(1, k+2)) # ii in [1, k+1]
    print(f"        alpha^0({j}+1) (k = {k}+1) = {val}")
    return val

def newtons_method(f, df, r_0, tol, N_max, debug=False):
    '''
    Approximate the solutions to f(x) = 0 with initial approximation x = r_0

    ### Parameters
        @f: single-variable scalar function
        @df: derivative of f
        @r_0: initial guess for root
        @tol: tolerance
        @N_max: maximum iterations to run

    ### Returns
        approximation (double), error_code (int)
    '''

    r_vec = np.zeros(N_max)
    r_vec[0] = r_0

    # Continue iterating until desired tolerance or max iterations reached
    for i in range(0, N_max):
        # Create next approximation
        r_prev = r_vec[i]
        r_curr = r_prev - (f(r_prev) / df(r_prev))
        r_vec[i+1] = r_curr

        # Check if absolute tolerance reached
        if np.abs(r_curr - r_prev) < tol:
            converged = True
            return (r_curr, converged)
    
    # If we reach this point, max iterations were reached
    return (r_curr, False)

def estimate_LTE(h_vec, t_vec, j, k, w0, wf):
    '''
    Estimates the local truncation error when using DASSL

    ### Parameters
    @h_vec: vector of h values
    @t_vec: vector of t values
    @j: time step
    @k: BDF order
    @w0: initial Newton's method guess
    @wf: result of Newton's method

    ### Returns
    double
    '''

    return abs((alpha(h_vec, t_vec, k+1, j) + alpha_s(k) - alpha_0(h_vec, t_vec, j, k)))\
        * abs(wf-w0)

def estimate_interp_err(h_vec, t_vec, j, k, w0, wf):
    '''
    Estimates the interpolation error when using DASSL

    ### Parameters
    @h_vec: vector of h values
    @t_vec: vector of t values
    @j: time step
    @k: BDF order
    @w0: initial Newton's method guess
    @wf: result of Newton's method

    ### Returns
    double
    '''

    return alpha(h_vec, t_vec, k, j) * abs(wf-w0)

# DASSL ########################################################################
################################################################################

def dassl_step(f, t_nodes, w_nodes, dw_nodes, j, h_j, h_fac, k_j, t_j, w_j, dw_j, newt_tol, debug=True):
    '''
    
    '''



    # Newton's method loop, we continue retrying with different initial
    # guesses (from trialing different stepsizes) until we are within
    # the specified tolerance error.
    newt_iter_count = 0
    while newt_iter_count < 10: # Terminate the whole algorithm if
                                # Newton's refuses to converge.
        # Calculate h_newt_j.
        h_j_newt = h_j / (h_fac**newt_iter_count)
        t_jp1_newt = t_j + h_j_newt

        # Create initial guesses for w_{j+1} and dw_{j+1}.
        if j == 0:
            # We don't have enough approximations for the interpolating
            # polynomial, so we'll use a Taylor expansion.
            w0_jp1_newt = w_j + h_j_newt * dw_j
            dw0_jp1_newt = dw_j
        else:
            # Use interpolating polynomial to create initial guess.
            w0_jp1_newt = eval_interp_poly(t_nodes, w_nodes, t_jp1_newt)
            dw0_jp1_newt = eval_interp_poly(t_nodes, dw_nodes, t_jp1_newt)

        # Create functions to run Newton's method on.
        ALPHA = -alpha_s(k_j) / h_j_newt
        BETA = dw0_jp1_newt - ALPHA*w0_jp1_newt
        f_newt = lambda w: f(t_jp1_newt, w, ALPHA*w + BETA)
        df_newt = lambda w: (f_newt(w)-f_newt(w-h_j_newt)) / h_j_newt

        if debug:
            print(f"      Beginning Newton's method with initial guesses w_{j+1} = {w0_jp1_newt}, dw_{j+1} = {dw0_jp1_newt},")
            print(f"                                                     h_{j} = {h_j_newt}, t_{j+1} = {t_jp1_newt}")
            print(f"                                                     ALPHA = {ALPHA}, BETA = {BETA}")

        # Run Newton's method.
        w_jp1_newt, converged = newtons_method(f_newt, df_newt, w0_jp1_newt, newt_tol, 4, debug=debug)
        dw_jp1_newt = ALPHA*w_jp1_newt + BETA
        newt_iter_count += 1

        if debug:
            print(f"        Newton's method converged? {converged}")
            print(f"        Approximations: w_{j+1} = {w_jp1_newt}, dw_{j+1} = {dw_jp1_newt}")

        # Do we need to retry the method?
        if converged: # No we do not.
            # Exit the Newton's method loop
            return h_j_newt, t_jp1_newt, w0_jp1_newt, w_jp1_newt, dw_jp1_newt
        else:
            # Continue the Newton's method loop
            newt_iter_count += 1
            continue
    
    if not converged:
        print("ERROR IN scalar_dassl: NEWTON'S FAILED TO CONVERGE AFTER 10 ITERATIONS")
        return

def scalar_dassl(f, t_0, t_f, y_0, dy_0, h_0, newt_tol, err_tol, debug=False):
    '''
    @f: f(t, y, y') = 0
    @t_0: left endpoint
    @t_f: right endpoint
    @y_0: y(t_0) = y_0
    @dy_0: y'(t_0) = dy_0
    @h_0: initial stepsize
    @newt_tol: successive approximation tolerance to use for terminating Newton's method
    @err_tol: error tolerance to use for LTE/interpolation error
    '''

    # Predicates
    assert t_0 < t_f
    assert h_0 > 0
    
    # Initialize vectors.
    t_vec = [t_0]
    w_vec = [y_0]
    dw_vec = [dy_0]
    h_vec = []
    k_vec = []
    w_vec_trials = [0]

    if dy_0 is None:
        print("ERROR IN scalar_dassl(): NO INITIAL DATA FOR dy_0 NOT YET IMPLEMENTED")
        return

    # Conditions for terminating the time stepping.
    interval_len = t_f - t_0
    interval_cov = t_0
    j = 0

    if debug:
        print(f"Running scalar_dassl() on [{t_0}, {t_f}] (length {interval_len}) with y({t_0}) = {y_0}, y'({t_0}) = {dy_0}")

    # Time stepping loop.
    # Each iteration (j) uses t_j, k_{j-1}, h_{j-1}, w_j, and dw_j to calculate
    # h_j and k_j in order to compute t_{j+1}, w_{j+1}, and dw_{j+1}.
    while interval_cov < interval_len:
        # Get values needed to compute t_{j+1}, w_{j+1}, and dw_{j+1}.
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
            k_jm1 = k_vec[-1] # t_j = t_{j-1} + h_{j-1}

        if debug:
            print(f"\n  Iteration ({j}): t_{j} = {t_j}, w_{j} = {w_j}, dw_{j} = {dw_j}")
            print(f"                 h_{j-1} = {h_jm1}, k_{j-1} = {k_jm1}")
        
        # Vectors of stepsizes, approximations, and BDF orders that have been
        # trialed for the current time step.
        t_jp1_trial_vec = []
        w_jp1_trial_vec = []
        dw_jp1_trial_vec = []
        h_j_trial_vec = []
        k_j_trial_vec = []
        accept_w_jp1 = False
        
        # Current time step (j) approximation loop.
        # Continue looping until our computed approximation meets certain
        # condtions.
        while not accept_w_jp1:
            trial_num = len(w_jp1_trial_vec)
            # Determine order of BDF method to use.
            if j == 0: # Use initial values.
                h_j = h_0
                k_j = 1
            else:
                k_j, h_j = calc_ord_stepsize(h_vec, t_vec, w_vec, h_jm1, j, k_jm1, len(w_jp1_trial_vec))

            # Get the previous (k+1) approximations to use as interpolant nodes
            # in the predictor polynomial.
            # TODO: maybe this should only be k nodes when creating w_1?
            t_pred_nodes = [t_vec[j-jj] for jj in range(k_j)] # jj in [0, k_j - 1]
            w_pred_nodes = [w_vec[j-jj] for jj in range(k_j)]
            dw_pred_nodes = [dw_vec[j-jj] for jj in range(k_j)]

            if debug:
                print(f"    Trial {trial_num}: using k_{j} = {k_j}, h_{j} = {h_j}")
                print(f"      Predictor polynomial nodes: {list(zip(t_pred_nodes, w_pred_nodes))}")

            # Trial with this step size
            h_j_trial, t_jp1_trial, w0_jp1_trial, w_jp1_trial, dw_jp1_trial =\
                dassl_step(f, t_pred_nodes, w_pred_nodes, dw_pred_nodes, j, h_j, 1/4, k_j, t_j, w_j, dw_j, newt_tol, debug=debug)
            h_j_trial_vec.append(h_j_trial)
            k_j_trial_vec.append(k_j)
            t_jp1_trial_vec.append(t_jp1_trial)
            w_jp1_trial_vec.append(w_jp1_trial)
            dw_jp1_trial_vec.append(dw_jp1_trial)

            if debug:
                print(f"      Trial {j} vectors:  h_{j} = {h_j_trial_vec}")
                print(f"                        k_{j} = {k_j_trial_vec}")
                print(f"                        t_{j+1} = {t_jp1_trial_vec}")
                print(f"                        w_{j+1} = {w_jp1_trial_vec}")
                print(f"                        dw_{j+1} = {dw_jp1_trial_vec}")

            # Now we check that this trial satisfies the error bounds
            h_vec_temp = h_vec.copy()
            h_vec_temp.append(h_j_trial)
            
            t_vec_temp = t_vec.copy()
            t_vec_temp.append(t_jp1_trial)

            if debug:
                print(f"      Estimating solution error with h = {h_vec_temp}")
                print(f"                                     t = {t_vec_temp}")
            
            LTE_j = estimate_LTE(h_vec_temp, t_vec_temp, j, k_j, w0_jp1_trial, w_jp1_trial)
            interp_err_j = estimate_interp_err(h_vec_temp, t_vec_temp, j, k_j, w0_jp1_trial, w_jp1_trial)

            if debug:
                print(f"        LTE error: {LTE_j}")
                print(f"        Interpolation error: {interp_err_j}")
            
            if max(LTE_j, interp_err_j) <= err_tol:
                # Step accepted!
                h_vec.append(h_j_trial)
                k_vec.append(k_j)
                t_vec.append(t_jp1_trial)
                w_vec.append(w_jp1_trial)
                dw_vec.append(dw_jp1_trial)
                interval_cov = t_jp1_trial
                j += 1
                accept_w_jp1 = True
                w_vec_trials.append(len(w_jp1_trial_vec))
                if debug:
                    print(f"      Trial {trial_num} solution ACCEPTED")
                    print(f"        h = {h_vec}")
                    print(f"        k = {k_vec}")
                    print(f"        t = {t_vec}")
                    print(f"        w = {w_vec}")
                    print(f"        dw = {dw_vec}")
            else:
                continue


