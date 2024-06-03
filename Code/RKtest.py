import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from Subroutines import RK_m, RK_m_var

def drive_RK(y, dy, a, b, alpha, N, methods, debug=False):    
    h = (b - a) / N
    t_vec = np.arange(a, b + h/10, h)
    y_vec = y(t_vec)

    for m in methods:
        t_m_vec, w_m_vec = RK_m(m, dy, a, b, alpha, N, debug=debug)
        err_m_vec = np.abs(y_vec - w_m_vec)
        plt.figure("Approximations")
        plt.plot(t_m_vec, w_m_vec, "-o")
        plt.figure("Error")
        plt.semilogy(t_m_vec, err_m_vec, "-o")
    plt.figure("Approximations")
    plt.plot(t_vec, y_vec)
    labels = [f"RK-{ord}" for ord in methods]
    labels.append("Actual")
    plt.figure("Error")
    labels = labels[0:len(labels)]
    plt.legend(labels)
    plt.show()

def drive_RK_var(y, dy, a, b, alpha, N, methods, tol, q_fac, debug=False):
    h_max = (b-a) / N
    t_vec = np.arange(a, b + h_max/10, h_max)
    y_vec = y(t_vec)

    fig_approx, ax_approx = plt.subplots()
    fig_error, ax_error = plt.subplots()
    for m in methods:
        t_m_vec, w_m_vec = RK_m(m, dy, a, b, alpha, N, debug=debug)
        err_m_vec = np.abs(y_vec - w_m_vec)
        ax_approx.plot(t_m_vec, w_m_vec, marker=".", label=f"RK-{m}")
        ax_error.semilogy(t_m_vec, err_m_vec, marker=".", label=f"RK-{m}")
    
    for m in methods:
        t_m_var_vec, w_m_var_vec = RK_m_var(m, dy, a, b, alpha, h_max, tol, q_fac, debug=debug)
        err_m_var_vec = np.abs(y(t_m_var_vec) - w_m_var_vec)
        ax_approx.plot(t_m_var_vec, w_m_var_vec, marker=".", label=f"Var. RK-{m}")
        ax_error.semilogy(t_m_var_vec, err_m_var_vec, marker=".", label=f"Var. RK-{m}")

    ax_approx.plot(t_vec, y_vec, label="Actual")
    ax_error.plot(t_vec, [tol] * len(t_vec), label="LTE tol")

    ax_approx.legend()
    ax_error.legend()
    plt.show()

dy_1 = lambda t, y: t + y
a = 1
b = 2
alpha = 1
N = 1
y_1 = lambda t: 3*np.exp(t-1) - t - 1
#drive_RK(y_1, dy_1, a, b, alpha, N, [3, 4])
N = 100
tol = 1e-1
q_fac = 0.9
debug = False
drive_RK_var(y_1, dy_1, a, b, alpha, N, [1, 2, 3], tol, q_fac, debug=False)
