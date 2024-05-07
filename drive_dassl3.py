# Imports ######################################################################
import numpy as np
import matplotlib.pyplot as plt
from dassl3 import start_dassl
################################################################################


def test_1():
    # dy_0 = y_0 + 2y_1
    # dy_1 = 3y_1 + 2y_1
    # y_0(0) = 0, y_1(0) = -4
    f_0 = lambda t, y, dy: 0*t + dy[0] - y[0] - 2*y[1]
    f_1 = lambda t, y, dy: 0*t + dy[1] - 3*y[0] - 2*y[1]
    F = lambda t, y, dy: np.array([f_0(t, y, dy), f_1(t, y, dy)])
    y0 = np.array([0, -4])
    dy0 = np.array([1, 1])

    # y_0(t) = (8/5)exp(-t) - (8/5)exp(4t)
    # y_1(t) = (-8/5)exp(-t) - (12/5)exp(4t)
    y_0 = lambda t: (8/5)*np.exp(-t) - (8/5)*np.exp(4*t)
    y_1 = lambda t: -(8/5)*np.exp(-t) - (12/5)*np.exp(4*t)

    t_0 = 0
    t_f = 3
    rel_tol = 1e-2
    abs_tol = 1e-2
    h_init = 0.1
    h_min = 1e-8
    h_max = 0.2
    max_ord = 6

    N = 100
    t_sol = np.linspace(t_0, t_f, N)
    y_0_sol = y_0(t_sol)
    y_1_sol = y_1(t_sol)

    
    t_ap, w, dw = start_dassl(F, t_0, t_f, y0, dy0, rel_tol, abs_tol, h_init,\
                            h_min, h_max, max_ord)
    
    plt.plot(t_sol, y_0_sol)
    plt.plot(t_sol, y_1_sol)
    plt.plot(t_ap, w[:, 0])
    plt.plot(t_ap, w[:, 1])
    plt.legend(["y_0(t)", "y_1(t)", "w_0(t)", "w_1(t)"])
    # plt.legend(["w_0(t)", "w_1(t)"])
    plt.show()
    
    

test_1()


