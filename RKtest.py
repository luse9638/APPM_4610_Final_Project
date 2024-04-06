import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from Subroutines import RK_m

def test_individual(y, dy, a, b, alpha, N):    
    t_vec = np.linspace(a, b, N)
    y_vec = y(t_vec)

    num_methods = 4
    plt.figure()
    for m in range(num_methods):
        t_m_vec, w_m_vec = RK_m(m+1, dy, a, b, alpha, N, debug=True)
        plt.plot(t_m_vec, w_m_vec)
    plt.plot(t_vec, y_vec)
    labels = [f"RK-{ord+1}" for ord in range(num_methods)]
    labels.append("Actual")
    plt.legend(labels)
    plt.show()

dy_1 = lambda t, y: t + y
a = 0
b = 5
alpha = 1
N = 500
y_1 = lambda t: 2*np.exp(t) - t - 1
test_individual(y_1, dy_1, a,b, alpha, N)
