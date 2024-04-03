import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from Subroutines import scalar_newtons

# Find root r = 1
def test_1():
    f = lambda x: np.exp(x) - x**2
    a = 0.669
    b = 0.0829
    print(scalar_newtons(f, a, b, -0.5, 1e-2, 1000))

test_1()




