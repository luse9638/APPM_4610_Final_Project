## Imports
##

import numpy as np
import scipy as sp
import math
import matplotlib.pyplot as plt
from Subroutines import mStepExplicitAB, RKm

def driver():
    f = lambda t, y: y**2 * t
    fact = lambda t: (-2) / (t**2 - 2)
    (t, w) = mStepExplicitAB(4, f, -1, 1, 2, 4, N=100, debug=True)
    plt.plot(t, fact(t))
    plt.plot(t, w)
    plt.show()
driver()