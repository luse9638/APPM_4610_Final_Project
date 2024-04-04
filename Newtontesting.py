import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from Subroutines import scalar_newtons, interpolate_at, interpolate_at_d

# Find root r = 1
def test_1():
    f = lambda x: np.exp(x) - x**2
    a = 0.669
    b = 0.0829
    print(scalar_newtons(f, a, b, -0.5, 1e-2, 1000))

def test_2():
    print(interpolate_at([0, 1, 2.5, 5, 7], [0, 3, 4, 2.5, 1], 6))
    print(interpolate_at_d([0, 1, 2.5, 5, 7], [0, 3, 4, 2.5, 1], 6))
test_2()




