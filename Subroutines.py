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

## DASSL
##

def interpolate_at(x, y, x_0, debug=False):
    """
    Constructs the Lagrange interpolating polynomial from x and y, and evaluates
    it at x_0

    ### Parameters
        @x: vector
        @y: vector
        @x_0: point to evaluate at

    ### Returns
        poly (polynomial), poly(x_0) (int)
    """

    poly = sp.interpolate.lagrange(x, y)
    if debug:
        print(poly)
    return (poly, poly(x_0))

def interpolate_at_d(x, y, x_0, debug=False):
    """
    Constructs the Lagrange interpolating polynomial from x and y, and evaluates
    its derivative at x_0

    ### Parameters
        @x: vector
        @y: vector
        @x_0: point to evaluate at

    ### Returns
        dpoly (polynomial), dpoly(x_0) (int)
    """

    poly = sp.interpolate.lagrange(x, y)
    dpoly = poly.deriv()
    if debug:
        print(dpoly)
    return (dpoly, dpoly(x_0))

def scalar_DASSL(f, t_0, t_f, alpha, alpha_prime, h_init, debug=False):
    """
    Solve ODE's of the form f(t, y(t), y'(t)) = 0 on the interval [t_0, t_f]

    ### Parameters
        @f: function of t, y, and y'
        @t_0: left endpoint of interval
        @t_f: right endpoint of interval
        @alpha: initial value for y, y(t_0) = alpha
        @alpha_prime: initial value for y', y'(t_0) = alpha_prime
        @h_init: Initial step size

    ### Returns
    """

    if debug:
        print(f"Running scalar_DASSL on interval [{t_0}, {t_f}] with y({t_0}) = {alpha}, y'({t_0}) = {alpha_prime}, and initial step size h = {h_init}")

    # Initialize t_vec, w_approx_vec, and dw_approx_vec vectors, w_approx_vec[j]
    # and dw[j] are approximations of solution w and its derivative dw at
    # t = t_j
    t_vec = [t_0]
    w_approx_vec = [alpha]
    dw_approx_vec = [alpha_prime]

    ## Mesh point loop
    for j in range(1, 1000): # Change later so we get right amount of meshpoints
        # TODO: recalculate order as needed
        ord = 1

        # TODO: recalculate h_j as needed
        h_j = h_init

        # Next t_j occurs at t_(j-1) + h
        # TODO: change so h adapts
        t_j = t_vec[-1] + h_j

        if debug:
            print("___________________________________________________________")
            print(f"| Order: {ord}, current step size: {h_j}")
            print(f"| Creating approximation w_{j} at t_{j} = {t_j}...")
        
        # w_j_init / dw_j_init represents the initial guess for w = w_j that
        # will later be iterated on by Newton's method
        w_j_init = 0
        dw_j_init = 0
        if j == 1:
            # Creating our first initial iterations for w_1 and dw_1 at 
            
            # y(t+h) ~= y(t) + h*y'(t) (first order Taylor)
            w_j_init = w_approx_vec[0] + h_j*dw_approx_vec[0]
            # y'(t+h) ~= y'(t)
            dw_j_init = dw_approx_vec[0]

            if debug:
                print(f"| | Creating initial guesses using Taylor: w_{j} = {w_j_init}, dw_{j} = {dw_j_init}")
            
            # When running Newton's method later we'll use y'(t) ~= a*y(t) + b,
            # a and b come from Hermite interpolation polynomial somehow... see
            # DASSL.jl
            a = 2 / h_j
            b = -dw_approx_vec[0] - ((2 * w_approx_vec[0]) / h_j) 
        else:
            # Not creating our first approximation, j != 1
            
            # We'll use interpolating polynomials with the nodes being the
            # previous (k+1) approximations to create initial iterations for
            # w_j and dw_j
            t_nodes = t_vec[-(ord+1):]
            w_nodes = w_approx_vec[-(ord+1):]
            poly, w_j_init = interpolate_at(t_nodes, w_nodes, t_j)
            dpoly, dw_j_init = interpolate_at_d(t_nodes, w_nodes, t_j)

            if debug:
                print(f"| | Creating initial guesses using polynomial: w_{j} = {w_j_init}, dw_{j} = {dw_j_init}")
                print(f"|   | Nodes used for polynomial: {list(zip(t_nodes, w_nodes))}")
                print(f"{poly}")

            # Again using y'(t) ~= a*y(t) + b, this time with different values
            # for a and b, no clue where these come from
            alphas = -sum(1/k for k in range(1, ord+1))
            a = -alphas / h_j
            b = dw_j_init - (a * w_j_init)
        
        if debug:
                print(f"| | Using dw_{j} ~= {a}*w_{j} + {b}")

        # Create the function f_newt to run Newton's method on, as solving 
        # f(t_j, y(t_j), y'(t_j)) = 0 improves our approximation w_j of y(t_j)
        # Here, we use the approximation y'(t_j) ~= a*y(t_j) + b using the a's
        # and b's calculated previously.
        f_newt = lambda w_j: f(t_j, w_j, a*w_j + b)
        # Also need the derivative of f_newt, calculated using first order
        # backwards difference:
        # ( f_newt(w_j) - f_newt(w_(j-1)) ) / ( t_j - t_(j-1) )
        df_newt = lambda w_j: (f_newt(w_j) - f_newt(w_j - h_j)) / h_j

        # Run Newton's method!
        # TODO: how to choose tolerance? how to choose max iterations? 4 is
        # used in DASSL.JL
        w_j, stat = scalar_newtons(f_newt, df_newt, w_j_init, 1e-4, 20, debug=debug)

        # Did we converge?
        if stat == 1:
            # We did not converge :/
            # TODO: how to update value for h and ord?
            return
        else:
            # We converged :)
            print(f"| | Newton's converged to w_{j} = {w_j}")
            w_approx_vec.append(w_j)
            dw_approx_vec.append(a*w_j + b)
            t_vec.append(t_j)
            print(f"| | t_vec: {t_vec}")
            print(f"| | w_approx_vec: {w_approx_vec}")
            print(f"| | dw_approx_vec: {dw_approx_vec}")
            print("----------------------------------------------------------\n")

        # Terminate once we've covered the entire interval
        if t_vec[-1] >= t_f:
            return(t_vec, w_approx_vec)



def scalar_newtons(f, df, r_0, tol, N_max, debug=False):
    """
    Approximate the solutions to f(x) = 0 with initial approximation x = r_0 
    and approximating f'(x) as a*f(x) + b

    ### Parameters
        @f: single-variable scalar function
        @df: derivative of f
        @r_0: initial guess for root
        @tol: tolerance
        @N_max: maximum iterations to run

    ### Returns
        approximation (double), error_code (int)
    """

    if debug:
        print(f"| | Running scalar_newtons with initial guess {r_0} with tolerance {tol}...")

    # Vector of approximations
    r_vec = np.zeros(N_max)
    r_vec[0] = r_0

    # Continue iterating until desired tolerance or max iterations reached
    for i in range(0, N_max - 1):
        # Create next approximation
        r_curr = r_vec[i] - (f(r_vec[i]) / df(r_vec[i]))
        if debug:
            print(f"| | | Iteration {i}: {r_curr}")
        r_vec[i+1] = r_curr

        # Check if absolute tolerance reached
        if np.abs(r_curr - r_vec[i]) < tol:
            err = 0
            return (r_curr, err)
    
    # If we reach this point, max iterations were reached
    return (r_curr, 1)

## Approximators
##

def RK_m(m, f, a, b, alpha, N=0, h=0, debug=False):
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
    
def m_step_expl_AB(m, f, a, b, alpha, order, N=0, h=0, debug=False):
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
    _, init_vec = RK_m(order, f, a, a + h*(m-1), alpha, m - 1, debug=debug)
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

def m_step_PCAB(m, f, a, b, alpha, order, N=0, h=0, debug=False):
    """ 
    Runs an (m-1)-step implicit Adams-Bashforth corrector method using an m-step explicit Adams-Bashforth predictor
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
        print("ERROR IN mStepPCAB(): NEED m > 0.")
        return
    if N == 0 and h == 0:
        print("ERROR IN mStepPCAB(): MUST SPECIFY VALUE FOR N or h.")
        return
   
    # Compute h and N, construct t_vec
    if N == 0:
        print("ERROR IN mStepPCAB(): NOT IMPLEMENTED CALCULATING h")
    elif h == 0:
        h = (b - a) / N
    t_vec = np.arange(a, b + h/10, h)

    # Create initial data using RK and initialize w_vec
    # Endpoint b is set so length of init_vec is equivalent to m
    if debug:
        print("Constructing initial data using RK...")
    _, init_vec = RK_m(order, f, a, a + h*(m-1), alpha, m - 1, debug=debug)
    w_vec = np.zeros(N + 1)
    for j in range(len(init_vec)):
        w_vec[j] = init_vec[j]

    # Compute b coefficients based on m for explicit predictor
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
        print("ERROR IN mStepPCAB(): THIS VALUE OF m NOT IMPLEMENTED")
        return
    
    # Compute c coefficients based on (m-1) for implicit corrector
    # w_(j+1) = w_j + h_m[c_(m)f(t_(j+1), w_(j+1)) + ... + c_(0)f(t_(i-m+1), w_(i-m+1))]
    mm = m-1
    c_vec = np.zeros(m)
    h_mm = 0
    if mm == 0:
        h_mm = h
        c_vec[0] = 1
    elif mm == 1:
        h_mm = h/2
        c_vec[0], c_vec[1] = 1, 1
    elif mm == 2:
        h_mm = h/12
        c_vec[0], c_vec[1], c_vec[2] = -1, 8, 5
    elif mm == 3:
        h_mm = h/24
        c_vec[0], c_vec[1], c_vec[2], c_vec[3] = 1, -5, 19, 9
    elif mm == 4:
        h_mm = h/720
        c_vec[0], c_vec[1], c_vec[2], c_vec[3], c_vec[4] = -19, 106, -264, 646, 251

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
        # Predict
        wp_j = w_vec[j] + h_m * (np.dot(f_eval_vec, b_vec))
        # Update function evalautions
        # Slide all elements to the left by 1
        f_eval_vec[:-1] = f_eval_vec[1:]
        # Replace last element with new predicted evaluation
        f_eval_vec[-1] = f(t_vec[j+1], wp_j)
        # Correct
        w_vec[j+1] = w_vec[j] + h_mm * (np.dot(f_eval_vec, c_vec))
        # Correct last function evaluation
        f_eval_vec[-1] = f(t_vec[j+1], w_vec[j+1])
        # if j+1 < N:
        #     # Slide all elements to the left by 1
        #     f_eval_vec[:-1] = f_eval_vec[1:]
        #     # Replace last element with new evaluation
        #     f_eval_vec[-1] = f(t_vec[j+1], w_vec[j+1])
        if debug:
            print(f"j = {j}")
            t_stat = [f"(t_{k}, w_{k})" for k in range(j-m+1, j+1)]
            print(f"Function evaluations at: {t_stat}")
            print(f"(t_{j+1}, w_{j+1}) = ({t_vec[j+1]}, {w_vec[j+1]})")
            print("\n")

    return t_vec, w_vec

def m_step_PCAB_var(m, f, a, b, alpha, order, eps, eps_factor, N=0, h=0, debug=False):
    """ 
    Runs a variable step size (m-1)-step implicit Adams-Bashforth corrector method using an m-step explicit Adams-Bashforth predictor
    Choose a value for h or N, but not both

    ### Parameters
        @m: Number of previous approximations to use in creation of next approximation
        @f: Differential equation y' = f(t, y)
        @a: left endpoint
        @b: right endpoint
        @alpha: y_0 = w_0 = alpha
        @order: order of RK method to use to create initial data
        @eps: desired tolerance of local truncation error
        @eps_factor: lower bound on local truncation error to determine when to change step size
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
        print("ERROR IN mStepPCAB(): NEED m > 0.")
        return
    if N == 0 and h == 0:
        print("ERROR IN mStepPCAB(): MUST SPECIFY VALUE FOR N or h.")
        return
   
    # Compute h and N, construct t_vec
    if N == 0:
        print("ERROR IN mStepPCAB(): NOT IMPLEMENTED CALCULATING h")
    elif h == 0:
        h = (b - a) / N
    t_vec = np.arange(a, b + h/10, h)

    # Create initial data using RK and initialize w_vec
    # Endpoint b is set so length of init_vec is equivalent to m
    if debug:
        print("Constructing initial data using RK...")
    _, init_vec = RK_m(order, f, a, a + h*(m-1), alpha, m - 1, debug=debug)
    w_vec = np.zeros(N + 1)
    for j in range(len(init_vec)):
        w_vec[j] = init_vec[j]

    # Compute b coefficients based on m for explicit predictor
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
        print("ERROR IN mStepPCAB(): THIS VALUE OF m NOT IMPLEMENTED")
        return
    
    # Compute c coefficients based on (m-1) for implicit corrector
    # w_(j+1) = w_j + h_m[c_(m)f(t_(j+1), w_(j+1)) + ... + c_(0)f(t_(i-m+1), w_(i-m+1))]
    mm = m-1
    c_vec = np.zeros(m)
    h_mm = 0
    if mm == 0:
        h_mm = h
        c_vec[0] = 1
    elif mm == 1:
        h_mm = h/2
        c_vec[0], c_vec[1] = 1, 1
    elif mm == 2:
        h_mm = h/12
        c_vec[0], c_vec[1], c_vec[2] = -1, 8, 5
    elif mm == 3:
        h_mm = h/24
        c_vec[0], c_vec[1], c_vec[2], c_vec[3] = 1, -5, 19, 9
    elif mm == 4:
        h_mm = h/720
        c_vec[0], c_vec[1], c_vec[2], c_vec[3], c_vec[4] = -19, 106, -264, 646, 251

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
    j = m-1
    eps_upper = eps
    eps_lower = eps*eps_factor
    # for j in range(m-1, N):
    while True:
        # Predict
        wp_j = w_vec[j] + h_m * (np.dot(f_eval_vec, b_vec))
        # Update function evalautions
        # Slide all elements to the left by 1
        f_eval_vec[:-1] = f_eval_vec[1:]
        # Replace last element with new predicted evaluation
        f_eval_vec[-1] = f(t_vec[j+1], wp_j)
        # Correct
        w_vec[j+1] = w_vec[j] + h_mm * (np.dot(f_eval_vec, c_vec))
        # Correct last function evaluation
        f_eval_vec[-1] = f(t_vec[j+1], w_vec[j+1])
        # Compute local truncation error
        # Do we need to update the step size?
        

        # if j+1 < N:
        #     # Slide all elements to the left by 1
        #     f_eval_vec[:-1] = f_eval_vec[1:]
        #     # Replace last element with new evaluation
        #     f_eval_vec[-1] = f(t_vec[j+1], w_vec[j+1])
        if debug:
            print(f"j = {j}")
            t_stat = [f"(t_{k}, w_{k})" for k in range(j-m+1, j+1)]
            print(f"Function evaluations at: {t_stat}")
            print(f"(t_{j+1}, w_{j+1}) = ({t_vec[j+1]}, {w_vec[j+1]})")
            print("\n")

    return t_vec, w_vec