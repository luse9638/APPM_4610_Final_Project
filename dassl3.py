# Imports ######################################################################
import numpy as np
import scipy as sp
################################################################################


# Constants ####################################################################
MACHINE_EPS = np.finfo(np.float64).eps
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

def interpolate_highest_derivative(x, y):
    if len(x) != len(y):
        raise ValueError("x and y have to be of the same size.")

    n = len(x)
    p = 0

    for i in range(n):
        Li = 1
        for j in range(n):
            if j == i:
                continue
            else:
                Li *= 1 / (x[i] - x[j])
        p += Li * y[i]

    return np.math.factorial(n - 1) * p

def numerical_jacobian(F, rel_tol, abs_tol):
    def numjac(t, y, dy, alpha):
        print(f"numerical_jacobian(): t = {t}")
        print(f"numerical_jacobian(): y = {y}")
        print(f"numerical_jacobian(): dy = {dy}")
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


# DASSL ########################################################################
def dassl_weights(v, rel_tol, abs_tol):
    return rel_tol*np.abs(v) + abs_tol

def dassl_norm(v, weights):
    v_div_weights = v / weights
    norm_result = np.linalg.norm(v_div_weights)
    return norm_result / np.sqrt(len(v))

def start_dassl(F, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, max_ord):
    """
    Initialize DASSL to approximate the solution to F(t, y, dy) = 0 on interval 
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
    @max_ord: upper bound on order.

    ## Returns
    (t, w, dw)
    """

    # Predicates.
    assert len(y_0) == len(dy_0)
    assert t_0 < t_f
    assert h_init >= h_min
    assert h_init <= h_max
    assert h_min < h_max
    assert max_ord >= 1

    # Initialize approximation vectors.
    interval_len = t_f - t_0
    max_steps = interval_len / h_min
    max_nodes = max_steps + 1
    n = len(y_0) # Number of dependent variables.

    #t = np.full(max_nodes, None)
    t = np.array([t_0])
    #w = np.full((max_nodes, n), None) # Each row is all variables at a time step, 
                                      # each column a variable at all time steps
    w = np.array([y_0])
    #dw = np.full((max_nodes, n), None)
    dw = np.array([dy_0])
    h = []
    k = []

    # Run DASSL.
    t, w, dw, k = dassl(F, t, w, dw, k, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, max_ord)

    return t, w, dw, k

def dassl(F, t, w, dw, k, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, max_ord):
    """
    Run the DASSL algorithm.

    ## Parameters
    @F: vector of n equations satisfying F_i(t, y, dy) = 0, 0 <= i <= n-1.
    @t: output vector of time nodes.
    @w: output vector of approximation nodes.
    @dw: output vector of derivative approximation nodes.
    @k: output vector of BDF orders.
    @t_0: left endpoint of interval.
    @t_f: right endpoint of interval.
    @y_0: vector of initial values satisfying y_i(t_0) = y_0_i.
    @dy_0: vector of initial values satisfying y'_i(t_0) = dy_0_i.
    @rel_tol: ???
    @abs_tol: ???
    @h_init: initial step size to use.
    @h_min: lower bound on step size.
    @h_max: upper bound on step size.
    @max_ord: upper bound on order.

    ## Returns
    """

    # Predicates.
    assert len(y_0) == len(dy_0)
    assert t_0 < t_f
    assert h_init >= h_min
    assert h_init <= h_max
    assert h_min < h_max
    assert max_ord >= 1

    # Initialize.
    interval_len = t_f - t_0
    max_steps = interval_len / h_min
    max_nodes = max_steps + 1
    n = len(y_0)

    t_out = np.array([t_0])
    #w_out = np.full((max_nodes, n), None) # Each row is all variables at a time 
                                          # step, each column a variable at all
                                          # time steps.
    w_out = np.array([y_0])
    #print(f"dassl(): w_out = {w_out}")
    #dw_out = np.full((max_nodes, n), None)
    dw_out = np.array([dy_0])
    h_out = []
    k_out = []

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

    while t_out[-1] < t_f:
        # Set initial step size and check for errors.
        h_min = max(4*MACHINE_EPS, h_min)
        h = min(h, h_max, t_f - t_out[-1])

        if h < h_min:
            raise ValueError(f"Step size too small (h = {h} at t = {t_out[-1]})")
        elif num_fail >= (-2/3) * np.log(MACHINE_EPS):
            raise ValueError(f"Too many ({num_fail}) steps in a row (h = {h} at t = {t_out[-1]})")
        
        # Set error weights and norm function.
        weights = dassl_weights(w_out[-1], rel_tol, abs_tol)
        norm_w = lambda v: dassl_norm(v, weights)

        # Take a time step!
        status, err, w_jp1, dw_jp1, jd = dassl_stepper(F, t_out, w_out, dw_out, h, jd, jacobian, weights, norm_w, ord, max_ord)

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
            
            # Temporarily add new step to t_out and w_out since they are needed
            # by new_step_order()
            t_out = np.append(t_out, t_out[-1] + h)
            w_out = np.vstack((w_out, w_jp1))

            # Determine new step size and order.
            r, new_ord = dassl_new_step_order(t_out, w_out, norm_w, err, num_fail, ord, max_ord)

            # Remove the temporary steps from before.
            t_out = t_out[:-1]
            w_out = w_out[:-1]

            # Change step size and order
            h *= r
            ord = new_ord
            continue
        else: # Step accepted!
            num_fail = 0

            # Save the results.
            t_out = np.append(t_out, t_out[-1] + h)
            w_out = np.vstack((w_out, w_jp1))
            dw_out = np.vstack((dw_out, dw_jp1))

            # Remove old results.
            if len(t_out) > ord+3:
                t_out = t_out[1:]
                w_out = w_out[1:]
                dw_out = dw_out[1:]
            
            # Add results to the output vectors.
            t = np.append(t, t_out[-1])
            w = np.vstack((w, w_out[-1]))
            dw = np.vstack((dw, dw_out[-1]))
            k = np.append(k, ord)

            # Determine new step size and order.
            r, new_ord = dassl_new_step_order(t_out, w_out, norm_w, err, num_fail, ord, max_ord)
            h *= r
            ord = new_ord

    return t, w, dw, k
            
def dassl_stepper(F, t, w, dw, h_next, jd, jacobian, weights, norm_w, ord, max_ord):
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
    @max_ord: ???

    ## Returns
    status, err, w_c, dw_c, jd
    """

    if w.ndim == 1:
        n = len(w) # Number of dependent variables.
    elif w.ndim == 2:
        n = len(w[0])
    else:
        raise ValueError(f"dassl_stepper(): w = {w} is not 1 or 2 dimensional")

    # Construct nodes for use in predictor polynomial.
    print(f"dassl_stepper(): t = {t}")
    t_nodes = t[-ord:]
    w_nodes = w[-ord:]
    print(f"dassl_stepper(): t_nodes = {t_nodes}, w_nodes = {w_nodes}")

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
        print(f"dassl_stepper(): Interpolating!")
        w_0 = interpolate_at(t_nodes, w_nodes, t_next)
        dw_0 = interpolate_derivative_at(t_nodes, w_nodes, t_next)
        alpha = -alpha_s / h_next
        beta = dw_0 - alpha*w_0
    
    print(f"dassl_stepper(): w_0 = {w_0}, dw_0 = {dw_0}")
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
        print(f"dassl_corrector(): jd.jac = {jd.jac}, F_newt = {F_newt}")
        f_newt = lambda w_c: -np.linalg.solve(jd.jac, F_newt(w_c))
        status, w_c = dassl_newton(f_newt, w_0, norm_w)
    else: # Use old Jacobian and see what happens.
        # Convergence speed up factor.
        c = 2*jd.alpha / (alpha_new+jd.alpha)

        # Run corrector.
        f_newt = lambda w_c: -c * np.linalg.solve(jd.jac, F_newt(w_c))
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
    w_jp1 = w_0 + delta

    if norm_1 < 100 * MACHINE_EPS * norm_w(w_0):
        return (0, w_jp1)

    # Iteration up to a maximum of MAXIT times
    for i in range(1, MAX_IT + 1):
        delta = f(w_jp1)
        norm_n = norm_w(delta)
        rho = (norm_n / norm_1) ** (1 / i)
        w_jp1 += delta

        if rho > 0.9:
            return (-1, w_0)  # Iteration failed to converge

        err = rho / (1 - rho) * norm_n
        if err < 1/3:
            return (0, w_jp1)  # Successful convergence

    # If the loop completes without returning, it failed to converge
    return (-1, w_0)

def dassl_new_step_order(t, w, norm_w, err, num_fail, ord, max_ord):
    """
    Compute a new step size and order.
    """

    # Predicates.
    assert len(t) == len(w)

    available_steps = len(t) # Includes t_{j+1}

    if num_fail >= 3:
        # We've probably already decreaed the step size a lot, so now we'll
        # reset the order.
        r = 1/4
        new_ord = 1
    elif num_fail == 1 and available_steps == 1 and err > 1:
        # w_j was accepted, first failure of w_{j+1}, reduce step size.
        r = 1/4
        new_ord = 1
    elif num_fail == 0 and available_steps == 2 and err < 1:
        # w_j was accepted, first attempt at w_{j+1}, increase order but keep
        # current step size.
        r = 1
        new_ord = 2
    elif available_steps < ord+2:
        # Still at the first few steps, can't do Taylor estimates yet, so these
        # adjustments are a little more crude.
        if num_fail == 0:
            # w_j was accepted, first attempt at w_{j+1}, increase order and
            # step size.
            r = (2*err + 1/10000) ** (-1 / (ord+1))
            #new_ord = ord # what DASSL.jl has
            new_ord = min(ord+1, max_ord)
        else:
            # w_{j+1} rejected, decrease step size and order.
            r = 1/4
            #new_ord = 1 # what DASSL.jl has
            new_ord = max(1, ord-1)
    else:
        # We have enough steps to do Taylor estimates.
        r, new_ord = dassl_new_step_order_taylor(t, w, norm_w, err, ord, max_ord)
        r = dassl_normalize_step(r, num_fail)
        if num_fail > 0: # Don't increase order
            new_ord = min(new_ord, ord)
    
    return r, new_ord

def dassl_new_step_order_taylor(t, w, norm_w, err, ord, max_ord):
    """
    Calculate step size and order.
    """
    errors = dassl_taylor_error_estimates(t, w, norm_w, ord)
    errors[ord-1] = err  
    errors_len = len(errors)

    # Are the errors geometrically monotone increasing or decreasing?
    range_idx = range(max(ord - 2, 1) - 1, min(errors_len, max_ord))
    errors_dec = all(np.diff([errors[i] for i in range_idx]) < 0)
    errors_inc = all(np.diff([errors[i] for i in range_idx]) > 0)

    if errors_len == ord + 1 and errors_dec:
        new_ord = min(ord + 1, max_ord)
    elif errors_len > 1 and errors_inc:
        new_ord = max(ord - 1, 1)
    else:
        new_ord = ord

    est = errors[new_ord-1]  # adjust for zero-based indexing

    # Initial guess for the new step size multiplier
    r = (2*est+0.0001) ** (-1/(new_ord+1))

    return r, new_ord

def dassl_taylor_error_estimates(t, w, norm_w, ord):
    """
    Estimates DASSL error.
    """
    n = len(t)
    h = t[-1] - t[-2]

    if n < ord + 2:
        raise ValueError(f"dassl_taylor_error_estimates() called with too few steps (n = {n}).")

    errors = np.zeros(ord)

    # Errors for order ord-2, ord-1, and ord.
    for i in range(max(ord - 2, 1), ord + 1):
        max_d = interpolate_highest_derivative(t[-(i + 1):], w[-(i + 1):])
        errors[i - 1] = h**(i + 1) * norm_w(max_d)

    # Error for order ord+1 if possible.
    if n >= ord + 3:
        h_n = np.diff(t[-(ord + 2):])
        if np.all(h_n == h_n[0]):
            max_d = interpolate_highest_derivative(t[-(ord + 2):], w[-(ord + 2):])
            errors = np.append(errors, h ** (ord + 2) * norm_w(max_d))

    return errors

def dassl_normalize_step(r, num_fail):
    if num_fail == 0:
        # w_j was accepted, first attempt at w_{j+1}
        if r >= 2:
            r_new = 2
        elif r < 1:
            r_new = max(1/2, min(r, 9/10))
        else:
            r_new = 1
    elif num_fail == 1:
        # Second attempt at w_{j+1}
        r_new = max(1/4, 9/10 * min(r, 1))
    else:
        r_new = 1/4
    
    return r_new
    
        






