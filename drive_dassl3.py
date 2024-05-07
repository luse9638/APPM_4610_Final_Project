# Imports ######################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from dassl3 import start_dassl
################################################################################

def test_dassl(F, y0, dy0, y, t_0, t_f, rel_tol, abs_tol, h_init, h_min, h_max, max_ord):
    N = 100
    t_sol = np.linspace(t_0, t_f, N)
    y_sol = np.array([y(t) for t in t_sol])  # Compute true solutions at t_sol points

    # Assuming start_dassl is the solver which must be defined appropriately
    # Placeholder values for demonstration:
    t_ap, w, _, k = start_dassl(F, t_0, t_f, y0, dy0, rel_tol, abs_tol, h_init, h_min, h_max, max_ord)
    k = np.append(0, k)

    # Plot solutions
    plt.figure("Solutions")
    for i in range(y_sol.shape[1]):  # Number of solution components
        plt.plot(t_sol, y_sol[:, i], label=f"y_{i}(t)")  # Corrected plotting
        plt.plot(t_ap, w[:, i], "-", label=f"w_{i}(t)")
        cmap = plt.cm.viridis  # You can choose any other colormap
        norm = mcolors.Normalize(vmin=np.min(k), vmax=np.max(k))
        scatter = plt.scatter(t_ap, w[:, i], c=k, cmap=cmap, norm=norm, edgecolor='black')
        if i == 0:
            cbar = plt.colorbar(scatter)
            cbar.set_label('BDF Order')
    plt.xlabel("t")
    plt.ylabel("Solution and Approximation")
    plt.title("Actual Solution y(t) vs. DASSL Approximation w(t)")
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

f_0 = lambda t, y, dy: dy[0] - 4*y[0] + 3*y[1] - t
f_1 = lambda t, y, dy: dy[1] - 2*y[0] + y[1] - np.exp(t)
F = lambda t, y, dy: np.array([f_0(t, y, dy), f_1(t, y, dy)])
y0 = np.array([0, 0])
dy0 = np.array([1, 1])

y_0 = lambda t: 3*np.exp(t)*t + t/2 + np.exp(t) - 9*np.exp(2*t)/4 + 5/4
y_1 = lambda t: 3*np.exp(t)*t + t - 3*np.exp(2*t)/2 + 3/2
y = lambda t: np.array([y_0(t), y_1(t)])

t_0 = 0
t_f = 5
rel_tol = 1e-3
abs_tol = 1e-4
h_init = 0.1
h_min = 1e-8
h_max = 1
max_ord = 15

test_dassl(F, y0, dy0, y, t_0, t_f, rel_tol, abs_tol, h_init, h_min, h_max, max_ord)


