import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator

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

# finding $e^{-20}$
N = 500 # no. of terms
x = -20 # evaluation point

N_arr1 = [] # array for terms
taylor_arr1 = [] # array for partial sum
error_arr1 = [] # array for error

# taylor expansion partial sum and error
for i in range(1, N):
    value = taylor_exp(i, x) # partial sum
    error = np.abs(value - math.exp(-20))/np.abs(math.exp(-20)) # error
    N_arr1.append(i) # update term
    taylor_arr1.append(value) # update partial sum
    error_arr1.append(error) # update error

    
print(f'Smallest error: ', min(error_arr1))

#parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 21
mpl.rcParams['axes.unicode_minus'] = False

# plotting the partial sum and error
fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(N_arr1, taylor_arr1, 'o', color='black')
plt.xlabel('$N$')
plt.ylabel(r'$\hat f_N(-20)$')
plt.ylim(1e-10, 1e8)
plt.text(160, 5e6, f"Final partial sum: {taylor_arr1[-1]:.16e}")
plt.text(160, 1e5, f"Smallest partial sum: {min(taylor_arr1):.16e}")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig('exp-20.pdf', dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(N_arr1, error_arr1, 'o', color='black')
plt.xlabel('$N$')
plt.ylabel(r'$E_N$')
plt.ylim(1e-1, 1e17)
plt.text(290, 5e15, f"Smallest error: {min(error_arr1):.8e}")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig('exp-20_error.pdf', dpi=1080)
plt.show()

# finding $\frac{1}{e^20}$
x = 20

N_arr2 = [] # array for terms
taylor_arr2 = [] # array for partial sum
error_arr2 = [] # array for error

# taylor expansion partial sum and error
for i in range(1, N):
    value = 1/(taylor_exp(i, x)) # partial sum
    error = np.abs(value - math.exp(-20))/np.abs(math.exp(-20)) # error
    N_arr2.append(i) # update term
    taylor_arr2.append(value) # update partial sum
    error_arr2.append(error) # update error

print(f'Smallest error: ', min(error_arr2))

# plotting the partial sum and error
fig, ax = plt.subplots(figsize=(12, 6))
ax.semilogy(N_arr2, taylor_arr2, 'o', color='black')
plt.xlabel('$N$')
plt.ylabel(r'${(\hat f_N(20))}^{-1}$')
plt.ylim(1e-9, 1e1)
plt.text(170, 1e0, f"Final partial sum: {taylor_arr2[-1]:.16e}")
plt.text(170, 1e-1, f"Smallest partial sum: {min(taylor_arr2):.16e}")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig('exp20.pdf', dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(12, 6))
plt.semilogy(N_arr2, error_arr2, 'o', color='black')
plt.xlabel('$N$')
plt.ylabel(r'$E_N$')
plt.ylim(1e-16, 1e10)
plt.text(290, 1e8, f"Smallest error: {min(error_arr2):.8e}")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig('exp20_error.pdf', dpi=1080)
plt.show()