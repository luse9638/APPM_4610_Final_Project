# Imports ######################################################################
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
################################################################################

# Classes ######################################################################
class Jac_Data:
    def __init__(self, a, jac):
        self.a = a
        self.jac = jac
    
    def __repr__(self):
        return f"JacData(a = {self.a}, jac = {self.jac})"
################################################################################

# Subroutines ##################################################################
def interpolate_at(x, y, x_0):
    """
    Returns the value of the interpolation polynomial at the point x0 using 
    Lagrange interpolation. x and y are numpy arrays containing the x and y 
    coordinates of the interpolation points, respectively. x0 is the point at
    which to evaluate the interpolation.
    """
    if len(x) != len(y):
        raise ValueError("x and y must be of the same size.")

    n = len(x)
    p = 0.0

    for i in range(n):
        # Start Li with 1 (equivalent to one(T) in Julia)
        Li = 1.0
        for j in range(n):
            if j != i:
                Li *= (x_0- x[j]) / (x[i] - x[j])

        p += Li * y[i]

    return p

def interpolate_derivative_at(x, y, x_0):
    """
    Returns the value of the derivative of the interpolation polynomial at the
    point x0. x and y are numpy arrays containing the x and y coordinates of the
    interpolation points, respectively. x0 is the point at which to evaluate the
    derivative of the interpolation.
    """
    if len(x) != len(y):
        raise ValueError("x and y must be of the same size.")

    n = len(x)
    p = 0.0

    for i in range(n):
        dLi = 0.0
        for k in range(n):
            if k == i:
                continue
            dLi1 = 1.0
            for j in range(n):
                if j == k or j == i:
                    continue
                dLi1 *= (x_0 - x[j]) / (x[i] - x[j])
            dLi += dLi1 / (x[i] - x[k])

        p += dLi * y[i]

    return p

def last_elements(arr, n=1):
    """
    Returns the last 'n' non-None elements from a 1D numpy array or the last 'n' non-None rows from a 2D numpy array.
    If there are fewer than 'n' non-None elements/rows, returns as many as are available.
    """

    # Predicates.
    assert isinstance(arr, np.ndarray), "Input must be a numpy array"
    assert n >= 1
    assert n <= len(arr)

    # Create a mask to filter out None elements or rows fully of None
    if arr.ndim == 1:
        # Filter out None values for 1D array
        mask = ~np.equal(arr, None)
    elif arr.ndim == 2:
        # Filter out rows that are fully None for 2D array
        mask = ~np.all(np.equal(arr, None), axis=1)
    else:
        raise ValueError("Array dimension higher than 2 is not supported")

    # Get indices where mask is True
    valid_indices = np.where(mask)[0]
    
    # Check if we have enough non-None elements/rows
    if len(valid_indices) < n:
        # If fewer than 'n', return all available non-None elements/rows
        return arr[valid_indices]
    else:
        # Return the last 'n' non-None elements/rows
        return arr[valid_indices[-n:]]
################################################################################

# Constants ####################################################################
MACHINE_EPS = np.finfo(np.float64).eps
################################################################################

# DASSL ########################################################################
def dassl_weights(v, rel_tol, abs_tol):
    return rel_tol*np.abs(v) + abs_tol

def dassl_norm(v, weights):
    v_div_weights = v / weights
    norm_result = np.linalg.norm(v_div_weights)
    return norm_result / np.sqrt(len(v))

def dassl_solve(F, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, ord_max):
    """
    Run DASSL to approximate the solution to F(t, y, dy) = 0 on interval 
    [t_0, t_f] using initial data y_0 and dy_0.

    ## Parameters
    @F: vector of n equations satisfying F_i(t, y, dy) = 0, 0 <= i <= n-1.
    @t_0: left endpoint of interval.
    @t_f: right endpoint of interval.
    @y_0: vector of n initial values satisfying y_i(t_0) = y_0_i.
    @dy_0: vector of n initial values satisfying y'_i(t_0) = dy_0_i.
    @rel_tol: ???
    @abs_tol: ???
    @h_init: initial step size to use.
    @h_min: lower bound on step size.
    @h_max: upper bound on step size.
    @ord_max: upper bound on order.

    ## Returns
    (w, dw)
    """

    # Predicates.
    assert len(y_0) == len(dy_0)
    assert t_0 < t_f
    assert h_init >= h_min
    assert h_init <= h_max
    assert h_min < h_max
    assert ord_max >= 1

    # Initialize approximation vectors.
    interval_len = t_f - t_0
    max_steps = interval_len / h_min
    max_nodes = max_steps + 1
    n = len(y_0) # Number of dependent variables.

    t = np.full(max_nodes, None)
    t[0] = t_0
    w = np.full((max_nodes, n), None) # Each row is all variables at a time step, 
                                      # each column a variable at all time steps
    w[0] = y_0
    dw = np.full((max_nodes, n), None)
    dw[0] = dy_0
    h = np.full(max_nodes, None)
    k = np.full(max_nodes, None)

    # Time-stepping loop.
    while last_elements(t) < t_f: # Continue time-stepping until we cover the interval.
        # TODO: call dassl_step() here
        pass

def dassl_step(F, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, ord_max):
    """
    Perform a single time-step of the DASSL algorithm.

    ## Parameters
    @F: vector of n equations satisfying F_i(t, y, dy) = 0, 0 <= i <= n-1.
    @t_0: left endpoint of interval.
    @t_f: right endpoint of interval.
    @y_0: vector of initial values satisfying y_i(t_0) = y_0_i.
    @dy_0: vector of initial values satisfying y'_i(t_0) = dy_0_i.
    @rel_tol: ???
    @abs_tol: ???
    @h_init: initial step size to use.
    @h_min: lower bound on step size.
    @h_max: upper bound on step size.
    @ord_max: upper bound on order.

    ## Returns
    """

    # Predicates.
    assert len(y_0) == len(dy_0)
    assert t_0 < t_f
    assert h_init >= h_min
    assert h_init <= h_max
    assert h_min < h_max
    assert ord_max >= 1

    # Initialize.
    interval_len = t_f - t_0
    max_steps = interval_len / h_min
    max_nodes = max_steps + 1
    n = len(y_0)

    t_out = np.full(max_nodes, None)
    t_out[0] = t_0
    w_out = np.full((max_nodes, n), None) # Each row is all variables at a time 
                                          # step, each column a variable at all
                                          # time steps.
    w_out[0] = y_0
    dw_out = np.full((max_nodes, n), None)
    dw_out[0] = dy_0
    h_out = np.full(max_nodes, None)
    k_out = np.full(max_nodes, None)

    ord = 1
    h = h_init

    num_rejected = 0
    num_fail = 0

    # TODO: run a single iteration of stepper here if initial derivative data is
    # missing.

    while last_elements(t_out) < t_f:
        # Set initial step size and check for errors.
        h_min = max(4*MACHINE_EPS, h_min)
        h = min(h, h_max, t_f - last_elements(t_out))

        if h < h_min:
            raise ValueError(f"Step size too small (h = {h} at t = {last_elements(t_out)})")
        elif num_fail >= (-2/3) * np.log(MACHINE_EPS):
            raise ValueError(f"Too many ({num_fail}) steps in a row (h = {h} at t = {last_elements(t_out)})")
        
        # Set error weights and norm function.
        weights = dassl_weights(last_elements(w_out), rel_tol, abs_tol)
        norm_w = lambda v: dassl_norm(v, weights)

        # TODO: call stepper here

def dassl_stepper(F, t, w, dw, h_next, jd, weights, norm_w, ord, ord_max):
    """
    Runs a single time-step of the DASSL algorithm.

    ## Parameters
    @F: vector of n equations satisfying F_i(t, y, dy) = 0, 0 <= i <= n-1.
    @t: vector of time nodes.
    @w: ???
    @dw: ???
    @h_next: ???
    @jd: ???
    @weights: ???
    @norm_w: ???
    @ord: ???
    @ord_max: ???

    ## Returns
    """

    n = len(w[0]) # Number of dependent variables.

    # Construct nodes for use in predictor polynomial.
    t_nodes = last_elements(t, ord)
    w_nodes = last_elements(w, ord)

    # Used in error calculations.
    #alpha_s = -sum(1/j for j in range(1, ord+2)) # j in [1, ord+1]
    alpha_s = -sum(1/j for j in range(1, ord+1)) # j in [1, ord]

    if len(w) == 1: # First time step.
        dw_0 = dw[0]
        w_0 = w[0] + h_next*dw_0

        # Alpha and beta values.
        alpha = 2 / h_next
        beta = -dw_0 - (2*w_0)/h_next
    else: # Not the first time step.
        # TODO: create approximations using predictor polynomial.
        pass

        
        



