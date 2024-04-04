import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from Subroutines import scalar_newtons, interpolate_at, interpolate_at_d, scalar_DASSL

# Find root r = 1
def test_1():
    f = lambda x: np.exp(x) - x**2
    a = 0.669
    b = 0.0829
    print(scalar_newtons(f, a, b, -0.5, 1e-2, 1000))

def test_2():
    print(interpolate_at([0, 1, 2.5, 5, 7], [0, 3, 4, 2.5, 1], 6))
    print(interpolate_at_d([0, 1, 2.5, 5, 7], [0, 3, 4, 2.5, 1], 6))

def test_3():
    f = lambda t, y, dy: 0*t + y - dy
    f_act = lambda t: t**2
    t_0 = 1
    t_f = 2
    alpha = 1
    alpha_prime = 2
    h = 0.05
    t_vec, w_vec = scalar_DASSL(f, t_0, t_f, alpha, alpha_prime, h, debug=True)

    for point in list(zip(t_vec, w_vec)):
        print(point)

test_3()




