import math
import numpy as np
import matplotlib.pyplot as plt

def taylor_exp(N, x):
    """
    Compute the N-term Taylor series expansion of e^x about 0, evaluated at x.
    Parameters
    ----------
    N : int
    Number of terms in the Taylor expansion.
    x : float
    Point at which to evaluate the expansion.
    Returns
    -------
    float
    Approximation of e^x using the first N terms of the series.
    """
    import math
    total = 0.0
    for n in range(N):
        total += (x**n) / math.factorial(n)
    return total

N = 1000
x = 20

N_arr = []
taylor_arr = []
error_arr = []

for i in range(N):
    value = 1/(taylor_exp(i, x) + 1e-30)
    error = np.abs(value - math.exp(-20))/np.abs(math.exp(-20))
    N_arr.append(i)
    taylor_arr.append(value)
    error_arr.append(error)

    
print(f'Smallest error: ', min(error_arr))

plt.semilogy(N_arr, error_arr, 'o')
    
