# Imports ######################################################################
import numpy as np
import matplotlib.pyplot as plt
from dassl3 import start_dassl
################################################################################

def test_dassl(F, y0, dy0, y, t_0, t_f, rel_tol, abs_tol, h_init, h_min, h_max, max_ord):
    N = 100
    t_sol = np.linspace(t_0, t_f, N)
    y_sol = np.array([y(t) for t in t_sol])  # Compute true solutions at t_sol points

    # Assuming start_dassl is the solver which must be defined appropriately
    # Placeholder values for demonstration:
    t_ap, w, _ = start_dassl(F, t_0, t_f, y0, dy0, rel_tol, abs_tol, h_init, h_min, h_max, max_ord)

    # Plot solutions
    plt.figure("Solutions")
    for i in range(y_sol.shape[1]):  # Number of solution components
        plt.plot(t_sol, y_sol[:, i], label=f"y_{i}(t) true")  # Corrected plotting
        plt.plot(t_ap, w[:, i], "-o", label=f"w_{i}(t) approx")
    plt.xlabel("t")
    plt.ylabel("Solution and Approximation")
    plt.title("Actual Solution vs. DASSL Approximation")
    plt.legend()

    # Plot error
    plt.figure("Error")
    w_err = np.zeros(len(t_ap))
    for i, t_i in enumerate(t_ap):
        y_t_i = y(t_i)  # Evaluate true solution at each approximated point
        w_err[i] = np.linalg.norm(w[i] - y_t_i, 2)
    plt.semilogy(t_ap, w_err, "-o")
    plt.plot(t_ap, [rel_tol] * len(t_ap))
    plt.plot(t_ap, [abs_tol] * len(t_ap))
    plt.xlabel("t")
    plt.ylabel("Norm of error")
    plt.legend(["Error", "Rel. Tol.", "Abs. Tol."])
    plt.title("Error of DASSL Approximation")

    plt.show()

f_0 = lambda t, y, dy: 0*t + dy[0] + y[1]
f_1 = lambda t, y, dy: 0*t + dy[1] - y[0]
F = lambda t, y, dy: np.array([f_0(t, y, dy), f_1(t, y, dy)])
y0 = np.array([1, 2])
dy0 = np.array([1, 1])

# y_0(t) = (8/5)exp(-t) - (8/5)exp(4t)
# y_1(t) = (-8/5)exp(-t) - (12/5)exp(4t)
y_0 = lambda t: np.cos(t) - 2*np.sin(t)
y_1 = lambda t: np.sin(t) + 2*np.cos(t)
y = lambda t: np.array([y_0(t), y_1(t)])

t_0 = 0
t_f = 5
rel_tol = 1e-4
abs_tol = 1e-5
h_init = 0.1
h_min = 1e-8
h_max = 0.2
max_ord = 6

test_dassl(F, y0, dy0, y, t_0, t_f, rel_tol, abs_tol, h_init, h_min, h_max, max_ord)


