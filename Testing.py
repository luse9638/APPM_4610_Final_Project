## Imports
##

import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from Subroutines import mStepExplicitAB, RKm, calc_error, mStepPCAB

def driver():
    f = lambda t, y: y**2 * t
    fact = lambda t: (-2) / (t**2 - 2)
    a = 0
    b = 1
    alpha = 1
    N = 50
    (t_AB, w_AB) = mStepExplicitAB(5, f, a, b, alpha, 4, N=N, debug=True)
    w_AB_err_vec, _ = calc_error(w_AB, fact(t_AB))
    (t_PCAB, w_PCAB) = mStepPCAB(4, f, a, b, alpha, 4, N=N, debug=True)
    w_PCAB_err_vec, _ = calc_error(w_PCAB, fact(t_PCAB))
    (t_RK4, w_RK4) = RKm(4, f, a, b, alpha, N=N, debug=True)
    w_RK4_err_vec, _ = calc_error(w_RK4, fact(t_RK4))
    plt.figure()
    plt.plot(t_AB, fact(t_AB))
    plt.plot(t_AB, w_AB)
    plt.plot(t_PCAB, w_PCAB)
    plt.plot(t_RK4, w_RK4)
    plt.title("Approximations")
    plt.legend(["Actual", "4-step AB", "4-step ABPC", "RK4"])
    plt.figure()
    plt.plot(t_AB, w_AB_err_vec)
    plt.plot(t_PCAB, w_PCAB_err_vec)
    plt.plot(t_RK4, w_RK4_err_vec)
    plt.title("Error")
    plt.legend(["4-step AB", "4-step ABPC", "RK4"])
    plt.show()


driver()