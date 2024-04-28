from dassl2 import *
import numpy as np

# Exponential test
def test_exp():
    c = 2
    y = lambda t: np.exp(c*t)
    
    f = lambda t, y, dy: dy - c*y + 0*t
    y_0 = 1
    dy_0 = c
    t_0 = 0
    t_f = 3

    newt_tol = 1e-1
    err_tol = 1
    h_0 = 0.1

    debug = True

    scalar_dassl(f, t_0, t_f, y_0, dy_0, h_0, newt_tol, err_tol, debug=debug)

test_exp()