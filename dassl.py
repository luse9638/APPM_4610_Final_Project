import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
import time
from Subroutines import interpolate_at, interpolate_at_d, scalar_newtons


def scalar_DASSL(f, t_0, t_f, y0, dy0, h_init, h_min, h_max, newt_tol, debug=False):
    """
    Solve ODE's of the form f(t, y(t), y'(t)) = 0 on the interval [t_0, t_f]

    ### Parameters
        @f: function of t, y, and y'
        @t_0: left endpoint of interval
        @t_f: right endpoint of interval
        @y0: initial value for y, y(t_0) = y0
        @dy0: initial value for y', y'(t_0) = dy0
        @h_init: Initial step size

    ### Returns
    """

    if debug:
        print(f"Running scalar_DASSL on interval [{t_0}, {t_f}] with y({t_0}) = {y0}, y'({t_0}) = {dy0}, and initial step size h = {h_init}")

    # Initialize vectors
    t_vec = [t_0]
    w_vec = [y0]
    dw_vec = [dy0]
    h_vec = [h_init]

    # Functions to use later
    def divided_diff(n, k):
        if k == 0:
            return w_vec[n]
        else:
            return (divided_diff(n, k-1) - divided_diff(n-1, k)) / (t_vec[n] - t_vec[n-k])
    
    def psi(m, j):
        assert(m >= 1)
        return t_vec[j] - t_vec[j-m]
    def alpha(m, j):
        assert(m >= 1)
        return h_vec[j] / psi(m, j)
    beta_1 = 1
    def beta(m, j):
        assert(m > 1)
        num = 1
        for mm in range(1, m): # [1, m-1]
            num *= psi(mm, j)
        denom = 1
        for mm in range(1, m): # [1, m-1]
            denom *= psi(mm, j-1)
        return num / denom
    def phi_1(j):
        return w_vec[j]
    def phi(m, j):
        assert (m > 1)
        result = 1
        for mm in range(1, m): # [1, m-1]
            result *= psi(mm, j)
        return result * divided_diff(j, m-1)
    def phi_star(m, j):
        assert(m >= 1)
        return beta(m, j+1) * phi(m, j)
    gamma_1 = 0
    def gamma(m, j):
        assert(m > 1)
        return gamma(m-1, j) + (alpha(m-1, j))/(h_vec[j])
    def alpha_bar(k):
        return -sum(1/j for j in range(1, k+1)) # [1, k]
    def alpha_tilde(k, j):
        return -sum(alpha(mm, j) for mm in range(1, k+1)) # [1, k]

    # Status variables
    # TODO: define these

    # Iteration count
    j = 0
    while True:
        # Values from previous iteration
        t_j = t_vec[-1]
        w_j = w_vec[-1]
        dw_j = dw_vec[-1]
        h_j = h_vec[-1]
        
        # TODO: calculate order
        k = 1

        # Create nodes for the predictor polynomial
        t_predict = t_vec[-(ord+1):]
        w_predict = w_vec[-(ord+1):]

        # Initialize h_jp1, may have to change later
        # This handles not overstepping our bounds
        h_jp1 = h_j
        h_jp1 = min(h_jp1, h_max, t_f - t_j)
        t_jp1 = t_j + h_jp1

        if debug:
            print("___________________________________________________________")
            print(f"| Order: {ord}, current step size: {h_jp1}")
            print(f"| Creating approximation w_{j+1} at t_{j+1} = {t_jp1}...")

        # Newton attempt count
        i = 1
        newton_converged = False
        # Continue retrying until iteration converges
        while not newton_converged:
            if debug:
                print(f"Newton attempt {i}")
            # Creating our first approximation, w_1: no need to interpolate, we use 
            # given data from the user
            if j == 0:
                # First order Taylor approximation
                w_jp1_init = w_j + h_j * dw_j
                # Assume derivative hasn't changed too much yet
                dw_jp1_init = dw_j
                
                # Create ALPHA and BETA
                #ALPHA = 2 / h_jp1
                ALPHA = -alpha_bar(k) / h_jp1
                BETA = dw_jp1_init - ALPHA*w_jp1_init
            # If not the first approximation, we gotta interpolate!
            else:
                predict_poly, w_jp1_init = interpolate_at(t_predict, w_predict, t_jp1)
                predict_dpoly, dw_jp1_init = interpolate_at_d(t_predict, w_predict, t_jp1)

                # Create ALPHA and BETA
                ALPHA = -alpha_bar(k) / h_jp1
                BETA = dw_jp1_init - ALPHA(w_jp1_init)
            if debug:
                print(f"| | Initial guesses: w_{j+1} = {w_jp1_init}, dw_{j+1} = {dw_jp1_init}")

            
            # Function to run Newton's method on
            f_newt = lambda w_jp1: f(t_jp1, w_jp1, ALPHA*w_jp1 + BETA)
            df_newt = lambda w_jp1: (f_newt(w_jp1) - f_newt(w_jp1 - h_jp1)) / h_jp1
            
            # Run Newton's method
            w_jp1, stat = scalar_newtons(f_newt, df_newt, w_jp1_init, newt_tol, 4, debug=debug)
            
            # Did not converge :/
            if stat == 1:
                # Decrease step size by 1/2
                h_jp1 = (1/2) * h_jp1
                t_jp1 = t_j + h_jp1
                ALPHA = -alpha_bar(k) / h_jp1
                BETA = dw_jp1_init - ALPHA(w_jp1_init)
                i += 1
                if debug:
                    print(f"| | Newton's did not converge, retrying with new step size h_jp1 = {h_jp1}")
            # Converged!
            else:
                w_vec.append(w_jp1)
                dw_vec.append(ALPHA*w_jp1 + BETA)
                t_vec.append(t_jp1)
                newton_converged = True
                if debug:
                    print(f"| | Newton's converged to w_{j} = {w_j}")
                    print(f"| | t_vec: {t_vec}")
                    print(f"| | w_vec: {w_vec}")
                    print(f"| | dw_vec: {dw_vec}")
                    print("----------------------------------------------------------\n")
                
                

        # Terminate once we've covered the entire interval
        if t_vec[-1] >= t_f:
            return(np.array(t_vec), np.array(w_vec))


        


