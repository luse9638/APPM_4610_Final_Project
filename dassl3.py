# Imports ######################################################################
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
################################################################################

# Classes ######################################################################
class Jac_Data:
    def __init__(self, alpha, jac):
        self.alpha = alpha
        self.jac = jac
    
    def __repr__(self):
        return f"JacData(a = {self.alpha}, jac = {self.jac})"
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

def numerical_jacobian(F, rel_tol, abs_tol):
    def numjac(t, y, dy, alpha):
        ep = np.finfo(float).eps  # machine epsilon
        h = 1 / alpha
        wt = dassl_weights(y, rel_tol, abs_tol)
        # delta for approximation of the jacobian
        edelta = np.diag(np.maximum(np.abs(y), np.maximum(np.abs(h * dy), wt)) * np.sqrt(ep))
        
        b = dy - alpha * y
        def f(y1):
            return F(t, y1, alpha * y1 + b)
        
        n = len(y)
        jac = np.zeros((n, n), dtype=float)
        for i in range(n):
            delta_i = np.zeros(n)
            delta_i[i] = edelta[i, i]
            jac[:, i] = (f(y + delta_i) - f(y)) / edelta[i, i]
        
        return jac
    
    return numjac
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

    jd = Jac_Data(0, np.full((n, n), None))
    jacobian = numerical_jacobian(F, rel_tol, abs_tol)

    ord = 1
    h = h_init

    num_rejected = 0
    num_fail = 0

    # Improve initial derivative guess.
    weights_init = dassl_weights(y_0, 1, 1)
    norm_init = lambda v: dassl_norm(v, weights_init)
    _, _, _, dw_out[0], _ = dassl_stepper(F, t_out, w_out, dw_out, 10*MACHINE_EPS, jd, jacobian, weights_init, norm_init, 1, 1)

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

        # Take a time step!
        status, err, w_n, dw_n, jd = dassl_stepper(F, t_out, w_out, dw_out, h, jd, jacobian, weights, norm_w, ord, ord_max)

        # Did the step work?
        if status == -1: # Newton iteration failed to converge. Reduce step size
                         # and retry.
            num_fail += 1
            num_rejected += 1
            h *= 1/4
            continue
        elif err > 1: # Error is too large, retry with a new step size and/or
                      # order.
            num_fail += 1
            num_rejected += 1
            # Temporarily add new step to t_out and w_out sicne they are needed
            # by new_step_order()
            

def dassl_stepper(F, t, w, dw, h_next, jd, jacobian, weights, norm_w, ord, ord_max):
    """
    Runs a single time-step of the DASSL algorithm.

    ## Parameters
    @F: vector of n equations satisfying F_i(t, y, dy) = 0, 0 <= i <= n-1.
    @t: vector of time nodes.
    @w: ???
    @dw: ???
    @h_next: ???
    @jd: ???
    @jacobian: ???
    @weights: ???
    @norm_w: ???
    @ord: ???
    @ord_max: ???

    ## Returns
    status, err, w_c, dw_c, jd
    """

    n = len(w[0]) # Number of dependent variables.

    # Construct nodes for use in predictor polynomial.
    t_nodes = last_elements(t, ord)
    w_nodes = last_elements(w, ord)

    # Next time step.
    t_next = t_nodes[-1] + h_next

    # Used in error calculations.
    #alpha_s = -sum(1/j for j in range(1, ord+2)) # j in [1, ord+1]
    alpha_s = -sum(1/j for j in range(1, ord+1)) # j in [1, ord]

    # Create initial guesses and calculate alpha and beta.
    if len(w) == 1: # First time step.
        # Just use 1st order taylor for initial guess.
        dw_0 = dw[0]
        w_0 = w[0] + h_next*dw_0
        alpha = 2 / h_next
        beta = -dw_0 - (2*w_0)/h_next
    else: # Not the first time step.
        # Use predictor polynomial for initial guess.
        w_0 = interpolate_at(t_nodes, w_nodes, t_next)
        dw_0 = interpolate_derivative_at(t_nodes, w_nodes, t_next)
        alpha = -alpha_s / h_next
        beta = dw_0 - alpha*w_0
    
    # Define function to run Newton's method on.
    F_newt = lambda w_c: F(t_next, w_c, alpha*w_c + beta)

    # Define new Jacobian function of F_newt.
    F_newt_jac_new = lambda: jacobian(t_next, w_0, dw_0, alpha)

    # Run the corrector.
    status, w_c, jd = dassl_corrector(F_newt, alpha, jd, F_newt_jac_new, w_0, norm_w)

    # For more error calculations.
    alpha_vec = np.zeros(ord+1)
    for i in range(ord):
        alpha_vec[i] = h_next / (t_next-t[-i - 1])
    if len(t) >= ord+1:
        t_0 = t[-ord - 1]
    elif len(t) >= 2:
        h_1 = t[1] - t[0]
        t_0 = t[0] - h_1
    else:
        t_0 = t[0] - h_next
    alpha_vec[ord] = h_next / (t_next-t_0)

    # Do the error calculations.
    alpha_0 = -np.sum(alpha_vec[:ord])
    M = max(alpha_vec[ord], abs(alpha_vec[ord] + alpha_s - alpha_0))
    err = norm_w(w_c-w_0) * M

    return status, err, w_c, alpha*w_c + beta, jd

def dassl_corrector(F_newt, alpha_new, jd, jac_new, w_0, norm_w):
    """
    Correct an initial DASSL guess.

    ## Parameters
    @F_newt: ???
    @alpha_new: ???
    @jd: ???
    @jac_new: ???
    @y0: ???
    @norm_w: ???

    ## Returns
    """

    # Compute new Jacobian if needed.
    if abs((jd.alpha-alpha_new) / (jd.alpha+alpha_new)) > 1/4:
        jd = Jac_Data(alpha_new, jac_new())
        
        # Run corrector.
        f_newt = lambda w_c: -np.linalg.solve(jd.jac, F_newt)
        status, w_c = dassl_newton(f_newt, w_0, norm_w)
    else: # Use old Jacobian and see what happens.
        # Convergence speed up factor.
        c = 2*jd.alpha / (alpha_new+jd.alpha)

        # Run corrector.
        f_newt = lambda w_c: -c * np.linalg.solve(jd.jac, F_newt)
        status, w_c = dassl_newton(f_newt, w_0, norm_w)

        if status == -1: # Corrector didn't converge with old Jacobian so we
                         # have to recompute :/.
            jd = Jac_Data(alpha_new, jac_new())
            f_newt = lambda w_c: -c * np.linalg.solve(jd.jac, F_newt)
            status, w_c = dassl_newton(f_newt, w_0, norm_w)
    
    return status, w_c, jd

def dassl_newton(f, w_0, norm_w, MAX_IT=4):
    # First prediction and check
    delta = f(w_0)
    norm_1 = norm_w(delta)
    w_n = w_0 + delta

    if norm_1 < 100 * MACHINE_EPS * norm_w(w_0):
        return (0, w_n)

    # Iteration up to a maximum of MAXIT times
    for i in range(1, MAX_IT + 1):
        delta = f(w_n)
        norm_n = norm_w(delta)
        rho = (norm_n / norm_1) ** (1 / i)
        w_n += delta

        if rho > 0.9:
            return (-1, w_0)  # Iteration failed to converge

        err = rho / (1 - rho) * norm_n
        if err < 1/3:
            return (0, w_n)  # Successful convergence

    # If the loop completes without returning, it failed to converge
    return (-1, w_0)
        



