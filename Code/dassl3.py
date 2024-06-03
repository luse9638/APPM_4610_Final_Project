# Imports ######################################################################
import numpy as np
import scipy as sp
################################################################################


# Constants ####################################################################
MACHINE_EPS = np.finfo(np.float64).eps # Machine epsilon.
################################################################################


# Classes ######################################################################
class Jac_Data:
    """
    Stores an alpha value and its associated jacobian matrix.

    ## Parameters
    @alpha: previous alpha value.
    @jac: previous jacobian matrix.
    """
    def __init__(self, alpha, jac):
        self.alpha = alpha
        self.jac = jac
    
    def __repr__(self): # Printing.
        return f"JacData(a = {self.alpha}, jac = {self.jac})"
################################################################################


# Subroutines ##################################################################
def interpolate_at(x, y, x_0):
    """
    Returns the value of the interpolation polynomial at the point x_0 using 
    Lagrange interpolation on nodes x and y.

    ## Parameters
    @x: numpy array
        vector of x values.
    @y: numpy array
        vector of y values.
    @x_0: numpy array
        point to evaluate at.

    ## Returns
    double
    """
    # Predicates.
    assert len(x) == len(y)

    n = len(x)
    p = 0.0
    for i in range(n):
        L_i = 1
        for j in range(n):
            if j != i:
                L_i *= (x_0-x[j]) / (x[i]-x[j])
        p += L_i*y[i]
    return p

def interpolate_derivative_at(x, y, x_0):
    """
    Returns the value of the derivative of the interpolation polynomial at the
    point x_0 using nodes x and y.

    ## Parameters
    @x: numpy array
        vector of x values.
    @y: numpy array
        vector of y values.
    @x_0: numpy array
        point to evaluate at.

    ## Returns
    double
    """
    # Predicates.
    assert len(x) == len(y)
    
    n = len(x)
    p = 0.0
    for i in range(n):
        dL_i = 0
        for k in range(n):
            if k == i:
                continue
            dL_i1 = 1
            for j in range(n):
                if j == k or j == i:
                    continue
                dL_i1 *= (x_0-x[j]) / (x[i]-x[j])
            dL_i += dL_i1 / (x[i]-x[k])
        p += dL_i*y[i]
    return p

def interpolate_highest_derivative(x, y):
    """
    For the interpolating polynomial with nodes x and y given by 
    p(x) = a_{k-1}*x^{k-1} + a_k*x^k + ... + a_1*x + a_0, return its highest
    derivative (the kth derivative), (k-1)!*a_{k-1}.

    ## Parameters:
    @x: numpy array
        vector of x values.
    @y: numpy array
        vector of y values.

    ## Returns
    double
    """
    # Predicates.
    assert len(x) == len(y)

    n = len(x)
    p = 0
    for i in range(n):
        L_i = 1
        for j in range(n):
            if j == i:
                continue
            else:
                L_i *= 1 / (x[i]-x[j])
        p += L_i*y[i]
    return np.math.factorial(n-1) * p

def numerical_jacobian(F, rel_tol, abs_tol):
    """
    Creates a function that approximates and evaluates the Jacobian matrix for 
    F(t, y, y') at a point using forward finite differences.

    ## Parameters
    @F: function
        F(t, y, y').
    @rel_tol: double
        relative tolerance for error norm.
    @abs_tol: double
        absolute tolerance for error norm.

    ## Returns
    function
        num_jac(t, y, dy, alpha)
    """
    
    def num_jac(t, y, dy, alpha):
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
    
    return num_jac
################################################################################


# DASSL ########################################################################
def dassl_weights(v, rel_tol, abs_tol):
    """
    Create weights to use in the error norm based on relative and absolute
    tolerances.

    ## Parameters
    @v: numpy array
        vector of values.
    @rel_tol: double
        relative tolerance to use in error norm.
    @abs_tol: double
        absolute tolerance to use in error norm.

    ## Returns
    numpy array
        vector of weights.
    """
    return rel_tol*np.abs(v) + abs_tol

def dassl_norm(v, weights):
    """
    Computes an error norm for a vector based on weights.

    ## Parameters
    @v: numpy array
        vector to compute norm of.
    @weights: numpy array
        vector of weights.
    
    ## Returns
    double
        error norm.
    """
    v_div_weights = v / weights
    norm_result = np.linalg.norm(v_div_weights)
    return norm_result / np.sqrt(len(v))

def start_dassl(F, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, max_ord):
    """
    Prepares to run DASSL to approximate the solution to F(t, y, dy) = 0 on
    the interval [t_0, t_f] using initial data y_0 and dy_0.

    ## Parameters
    @F: function
        vector of n functions satisfying F_i(t, y, dy) = 0, 0 <= i <= n-1.
    @t_0: double
        left endpoint of interval.
    @t_f: double
        right endpoint of interval.
    @y_0: numpy array
        vector of n initial values satisfying y_i(t_0) = y_0_i.
    @dy_0: numpy array
        vector of n initial values satisfying y'_i(t_0) = dy_0_i.
    @rel_tol: double
        relative tolerance to use in norm.
    @abs_tol: double
        absolute tolerance to use in norm.
    @h_init: double
        initial step size to use.
    @h_min: double
        lower bound on step size.
    @h_max: double
        upper bound on step size.
    @max_ord: int
        upper bound on order.

    ## Returns
    t, w, dw, k
        t: numpy array
            vector of time steps.
        w: numpy array
            vector of approximations to y.
        dw: numpy array
            vector of approximations to y'.
        k: numpy array
            vector of BDF orders used.
    """

    # Predicates.
    assert len(y_0) == len(dy_0)
    assert t_0 < t_f
    assert h_init >= h_min
    assert h_init <= h_max
    assert h_min < h_max
    assert max_ord >= 1

    # Initialize approximation vectors.
    t = np.array([t_0])
    w = np.array([y_0]) # Each row stores all variables at a specific time step,
                        # each column stores a single variable at all time steps.
    dw = np.array([dy_0])
    k = []

    # Run DASSL.
    t, w, dw, k = dassl(F, t, w, dw, k, t_0, t_f, y_0, dy_0, rel_tol, abs_tol,\
                        h_init, h_min, h_max, max_ord)

    return t, w, dw, k

def dassl(F, t, w, dw, k, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, max_ord):
    """
    Run the DASSL algorithm.

    ## Parameters
    @F: function
        vector of n functions satisfying F_i(t, y, dy) = 0, 0 <= i <= n-1.
    @t: numpy array
        vector to store time steps.
    @w: numpy array
        vector to store approximations to y.
    @dw: numpy array
        vector to store approximations to y'.
    @k: array
        vector to BDF orders.
    @t_0: double
        left endpoint of interval.
    @t_f: double
        right endpoint of interval.
    @y_0: numpy array
        vector of n initial values satisfying y_i(t_0) = y_0_i.
    @dy_0: numpy array
        vector of n initial values satisfying y'_i(t_0) = dy_0_i.
    @rel_tol: double
        relative tolerance to use in norm.
    @abs_tol: double
        absolute tolerance to use in norm.
    @h_init: double
        initial step size to use.
    @h_min: double
        lower bound on step size.
    @h_max: double
        upper bound on step size.
    @max_ord: int
        upper bound on order.

    ## Returns
    t, w, dw, k
        t: numpy array
            vector of time steps.
        w: numpy array
            vector of approximations to y.
        dw: numpy array
            vector of approximations to y'.
        k: numpy array
            vector of BDF orders used.
    """

    # Initialize approximation vectors.
    n = len(y_0)

    t_save = np.array([t_0])
    w_save = np.array([y_0]) # Each row stores all variables at a specific time step,
                            # step, each column stores a single variable at all
                            # time steps.
    dw_save = np.array([dy_0])

    # Used to store previous jacobian and alpha value.
    jd = Jac_Data(0, np.full((n, n), None))

    # Create the jacobian function.
    jacobian = numerical_jacobian(F, rel_tol, abs_tol)

    # Initial order and step size.
    ord = 1
    h = h_init
    
    # How many steps have failed and how many have been rejected.
    num_rejected = 0
    num_fail = 0

    # Improve initial derivative guess by running a single step.
    weights_init = dassl_weights(y_0, 1, 1)
    norm_init = lambda v: dassl_norm(v, weights_init)
    _, _, _, dw_save[0], _ = dassl_stepper(F, t_save, w_save, dw_save,\
                                          10*MACHINE_EPS, jd, jacobian,\
                                            weights_init, norm_init, 1, 1)

    # Time stepping loop.
    while t_save[-1] < t_f:
        # Set initial step size and check for errors.
        h_min = max(4*MACHINE_EPS, h_min)
        h = min(h, h_max, t_f - t_save[-1])
        if h < h_min:
            raise ValueError(f"ERROR IN dassl(): STEP SIZE TOO SMALL (h = {h} AT t = {t_save[-1]})")
        elif num_fail >= (-2/3) * np.log(MACHINE_EPS):
            raise ValueError(f"ERROR IN dassl(): TOO MANY FAILED STEPS ({num_fail}) IN A ROW (h = {h} at t = {t_save[-1]})")
        
        # Set error weights and norm function.
        weights = dassl_weights(w_save[-1], rel_tol, abs_tol)
        norm_w = lambda v: dassl_norm(v, weights)

        # Take a time step!
        status, err, w_jp1, dw_jp1, jd = dassl_stepper(F, t_save, w_save, dw_save,\
                                                       h, jd, jacobian, weights,\
                                                        norm_w, ord, max_ord)

        # Check if Newton's converged or not.
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
            
            # Temporarily add the (rejected) new step to t_save and w_save since 
            # it is needed by new_step_order().
            t_save = np.append(t_save, t_save[-1] + h)
            w_save = np.vstack((w_save, w_jp1))

            # Determine new step size and order.
            r, new_ord = dassl_new_step_order(t_save, w_save, norm_w, err, num_fail, ord, max_ord)

            # Remove the temporary (rejected) step that was added from before.
            t_save = t_save[:-1]
            w_save = w_save[:-1]

            # Change step size and order.
            h *= r
            ord = new_ord
            continue
        else: # Step accepted!
            num_fail = 0

            # Save the results.
            t_save = np.append(t_save, t_save[-1] + h)
            w_save = np.vstack((w_save, w_jp1))
            dw_save = np.vstack((dw_save, dw_jp1))

            # Remove old results.
            if len(t_save) > ord+3:
                t_save = t_save[1:]
                w_save = w_save[1:]
                dw_save = dw_save[1:]
            
            # Add results to the output vectors.
            t = np.append(t, t_save[-1])
            w = np.vstack((w, w_save[-1]))
            dw = np.vstack((dw, dw_save[-1]))
            k = np.append(k, ord)

            # Determine new step size and order.
            r, new_ord = dassl_new_step_order(t_save, w_save, norm_w, err, num_fail, ord, max_ord)
            h *= r
            ord = new_ord

    return t, w, dw, np.array(k)
            
def dassl_stepper(F, t, w, dw, h_next, jd, jacobian, weights, norm_w, ord, max_ord):
    """
    Runs a single time-step of the DASSL algorithm.

    ## Parameters
    @F: function
        vector of n functions satisfying F_i(t, y, dy) = 0, 0 <= i <= n-1.
    @t: numpy array
        vector of previous time steps.
    @w: numpy array
        vector of previous approximations to y.
    @dw: numpy array
        vector of previous approximations to y'.
    @h_next: double
        time step to use.
    @jd: Jac_Data
        previous alpha and jacobian matrix.
    @jacobian: function
        computes current jacobian matrix.
    @weights: numpy array
        vector of weights to use for error norm.
    @norm_w: function
        function to compute error norm of vector.
    @ord: int
        order to use.
    @max_ord: int
        upper bound on order.

    ## Returns
    status, err, w_c, dw_c, jd
        status: int
            -1 if Newton's method failed to converge, 0 otherwise.
        err: double
            estimate of LTE / interpolation error.
        w_c: numpy array
            approximation to y at current time step.
        dw_c: numpy array
            approximation to y' at current time step.
        jd: Jac_Data
            Jacobian data used to create approximation.
    """
    # Construct nodes for use in predictor polynomial.
    print(f"dassl_stepper(): t = {t}")
    t_nodes = t[-ord:]
    w_nodes = w[-ord:]
    print(f"dassl_stepper(): t_nodes = {t_nodes}, w_nodes = {w_nodes}")

    # Next time step.
    t_next = t_nodes[-1] + h_next

    # Used in error calculations.
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
    Correct an initial DASSL guess using Newton's method.

    ## Parameters
    @F_newt: function
        function to run Newton's method on.
    @alpha_new: double
        alpha value to use.
    @jd: Jac_Data
        previous alpha value and jacobian matrix.
    @jac_new: function
        function to compute current jacobian matrix.
    @w_0: numpy array
        initial guess.
    @norm_w: function
        function to compute error norm of vector.

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
    
        






