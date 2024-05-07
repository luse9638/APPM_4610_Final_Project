# Imports ######################################################################
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
################################################################################

# Subroutines ##################################################################
def last_element(arr):
    """
    Given an array with unitialized entries (None) at the end, return the last
    element that is not None.
    """

    # Predicate.
    assert isinstance(arr, np.ndarray)

    arr_mask = ~np.equal(arr, None)
    indices = np.where(arr_mask)[0]

    if indices.size > 0:
        return arr[indices[-1]]
    else:
        return None
################################################################################

# DASSL ########################################################################

def dassl_solve(F, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, k_max):
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
    @k_max: upper bound on order.

    ## Returns
    (w, dw)
    """

    # Predicates.
    assert len(y_0) == len(dy_0)
    assert t_0 < t_f

    # Initialize approximation vectors.
    num_eqns = len(F)
    interval_len = t_f - t_0
    max_steps = interval_len / h_min
    max_nodes = max_steps + 1

    t = np.full(max_nodes, None)
    t[0] = t_0
    w = np.full((num_eqns, max_nodes), None)
    w[0] = y_0
    dw = np.full((num_eqns, max_nodes), None)
    dw[0] = dy_0
    h = np.full(max_nodes, None)
    k = np.full(max_nodes, None)

    # Time-stepping loop.
    while t[-1] < t_f: # Continue time-stepping until we cover the interval.
        # TODO: call dassl_step() here
        pass

def dassl_step(F, t_0, t_f, y_0, dy_0, rel_tol, abs_tol, h_init, h_min, h_max, k_max):
    """
    Perform a single time-step of the DASSL algorithm.

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
    @k_max: upper bound on order.

    ## Returns
    """

