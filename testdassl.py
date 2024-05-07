from dassl2 import *
import numpy as np
import matplotlib.pyplot as plt

# Exponential test
def test_dassl(y, f, y_0, dy_0, t_0, t_f, h_0, newt_tol, err_tol, debug):
    t_act = np.linspace(t_0, t_f, 500)
    y_act = y(t_act)
    
    t_vec, w_vec = scalar_dassl(f, t_0, t_f, y_0, dy_0, h_0, newt_tol, err_tol, debug=debug)
    plt.figure()
    plt.plot(t_act, y_act)
    plt.plot(t_vec, w_vec, "-o")
    plt.legend(["Actual", "DASSL"])
    plt.show()



# y(t) = e^t^2/2
c = 2
y = lambda t: np.exp((1/2) * t**2)
f = lambda t, y, dy: dy - t*y
y_0 = 1
dy_0 = 0
t_0 = 0
t_f = 1
newt_tol = 1e-2
err_tol = 1e-2
h_0 = 0.1
debug = True
test_dassl(y, f, y_0, dy_0, t_0, t_f, h_0, newt_tol, err_tol, debug)
