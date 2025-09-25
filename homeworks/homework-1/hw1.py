import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from sympy import symbols, Eq, plot
#import statsmodels.api as sm
#from sklearn.linear_model import RANSACRegressor, LinearRegression

# ===== 1. Vandermonde matrix =====

#function to construct Vandermonde matrix
def vandermonde(time_sample, degree):
    matrix_col = []
    for m in range(degree+1):
        matrix_col.append(time_sample ** m)
    matrix = pd.concat(matrix_col, axis=1)
    return matrix

# ===== 2. Least Squared Regression ===== 
    
#input arguments
filename = "hw1.csv"
degree = [1, 3, 9]

x_list = {} #co-efficients
lse_list = {} #least squares error (LSE)
mse_list = {} #mean squared error (MSE)

col1 = pd.read_csv(filename, usecols=[0], header=None); #time samples
col2 = pd.read_csv(filename, usecols=[1], header=None); #target vector

for i in degree:
    vandermonde_ = vandermonde(col1, i)
    x = (np.linalg.inv(vandermonde_.T @ vandermonde_) @ vandermonde_.T @ col2).to_numpy().flatten() #$\omega = (A^TA)^{-1}A^T\vec{y}$
    y = col2.to_numpy().flatten()
    lse = np.sum((abs(y - vandermonde_ @ x))**2) #$\sum{{\left|\vec{y} - A\omega \right|}^2}$
    mse = np.mean((abs(y - vandermonde_ @ x))**2) #$\frac{1}{N}\sum{{\left|\vec{y} - A\omega \right|}^2}$
    x_list[i] = x #update co-efficient list
    lse_list[i] = lse #update LSE list
    mse_list[i] = mse #update MSE list
print('Co-efficients:', x_list)
print('LSE:', lse_list)
print('MSE:', mse_list)

#function to construct polynomial equation from co-efficients
def linear_reg(coeffs):
    x_ = symbols("x");
    y_ = 0;
    for i, a in enumerate(coeffs):
        y_ += a * x_ ** i;
    return Eq(symbols('y'), y_);

eq_list = {} #equations
for i in x_list:
    coeffs = x_list[i]
    print(f"Coefficients: {coeffs}")
    equation = linear_reg(coeffs)
    eq_list[i] = equation
print(eq_list)

#parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 22
mpl.rcParams['axes.unicode_minus'] = False

#plotting given data and least squares regression line for each polynomial degree
for i, equation in eq_list.items():
    fig, ax = plt.subplots(figsize=(8, 6))
    plt.scatter(col1, col2, color='red', label='Data')
    eq_plot = plot(equation.rhs, (symbols("x"), col1.to_numpy().flatten().min(), 
                                  col1.to_numpy().flatten().max()), show=False)

    eq_plot[0].line_color = 'black'
    eq_plot[0].line_width = 3
    eq_plot[0].label = f'Polynomial degree-{i}'

    for line in eq_plot:
        ax.plot(*line.get_points(), color=line.line_color,
                linewidth=line.line_width, linestyle='-', 
                label=line.label)

    plt.xlabel('$t$')
    plt.ylabel('$y$')
    plt.xlim(col1.to_numpy().flatten().min(), col1.to_numpy().flatten().max())
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f'regress_deg{i}.pdf', dpi=1080)
    plt.show()

#plotting LSE and MSE for each polynomial degree
lse = [lse_list[i] for i in degree]
mse = [mse_list[i] for i in degree]
fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(degree, lse, color='blue', linestyle='-', linewidth=3, marker='o', markersize=8, label='$LSE$')
ax.semilogy(degree, mse, color='red', linestyle='-', linewidth=3, marker = 's', markersize=8, label='$MSE$')
#plt.xlim(0, 10)
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.ylim(10**(-5), 10**0)
plt.xlabel('$M$')
plt.ylabel('$LSE$, $MSE$')
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig('lse_mse.pdf', dpi=1080)
plt.show()

# ===== 3. Training & testing data MSE =====

#preparing data
data = np.column_stack((col1.to_numpy().flatten(), col2.to_numpy().flatten()))
np.random.seed(32) #for reproduciblity
np.random.shuffle(data) #shuffling positions
split = int(0.8 * len(data)) #80% training, 20% testing
train, test = data[:split], data[split:]
t_train, y_train = train[:,0], train[:,1]
t_test, y_test = test[:,0], test[:,1]

mse_train_list = {} #training MSE
mse_test_list = {} #testing MSE
cond_no_train_list = {} #training condition no.

for i in degree:
    vndrmnd_train = vandermonde(pd.DataFrame(t_train), i) #Vandermonde matrix for training data
    vndrmnd_test = vandermonde(pd.DataFrame(t_test), i) #Vandermonde matrix for testing data
    x_train = (np.linalg.inv(vndrmnd_train.T @ vndrmnd_train) @ vndrmnd_train.T @ pd.DataFrame(y_train)).to_numpy().flatten() #$\omega$ using training data
    y_train_predict = vndrmnd_train @ x_train #predicted traget vector for training data
    y_test_predict = vndrmnd_test @ x_train #predicted traget vector for testing data
    mse_train = np.mean((abs(y_train_predict - y_train))**2)
    mse_train_list[i] = mse_train #update traing MSE list
    mse_test = np.mean((abs(y_test_predict - y_test))**2)
    mse_test_list[i] = mse_test #update testing MSE list
    cond_no_train = np.linalg.cond(vndrmnd_train, 2) #$\|A\|_2 \|A^+\|_2$
    cond_no_train_list[i] = cond_no_train #update condition no. list
    
print('MSE (training):', mse_train_list)
print('MSE (testing):', mse_test_list)
print('Condition no. (training):', cond_no_train_list)

# ===== 4. Condition number =====

N = [10, 40, 100] #subsample size
cond_no_list = {} #condition no.

for i in degree:
    cond_no_list[i] = {}
    for j in N:
        if j > len(data):
            continue
        subsample = np.random.choice(len(data), size=j, replace=False)
        data_ = data[subsample]
        vndrmnd = vandermonde(pd.DataFrame(data_[:,0]), i)
        #print(np.linalg.inv(vndrmnd))
        cond_no = np.linalg.cond(vndrmnd, 2) #$\|A\|_2 \|A^+\|_2$
        cond_no_list[i][j] = cond_no #update condition no. list

print('Condition no.:')
for i in cond_no_list:
    print(f'degree-{i}:')
    for j in cond_no_list[i]:
        print(f'N = {j}: {cond_no_list[i][j]: .6e}')

#plotting condition no. for different subsample sizes
fig, ax = plt.subplots(figsize=(10, 6))
mrkr = ['o', 's', '^']
ls = ['-', '--', '-.']
clr = ['blue', 'red', 'forestgreen']
for sym, j in enumerate(N):
    x = []
    y = []
    for i in degree:
        if j in cond_no_list[i]:
            x.append(i)
            y.append(cond_no_list[i][j])
    ax.semilogy(x, y, color=clr[sym%len(clr)], linestyle=ls[sym%len(ls)], linewidth=3, marker=mrkr[sym%len(mrkr)], markersize=8, label=f'$N = {j}$')
plt.xlabel(r'$M$')
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.ylabel(r'$\kappa$')
plt.ylim(10**(0), 10**(10))
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig('cond_no.pdf', dpi=1080)
plt.show()
        
# ===== 5. MSE and condition number plots =====
        
#plotting MSE for training and testing data
mse_train = np.array([mse_train_list[i] for i in degree])
mse_test = np.array([mse_test_list[i] for i in degree])

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(degree, mse_train, color='blue', linestyle='-', linewidth=3, marker='o', markersize=8, label='MSE (training)')
ax.semilogy(degree, mse_test, color='red', linestyle='--', linewidth=3, marker='s', markersize=8, label='MSE (testing)')
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.ylim(10**(-5), 10**(-2))
plt.xlabel(r'$M$')
plt.ylabel(r'$MSE$')
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig('mse.pdf', dpi=1080)
plt.show()

#plotting condition no. for training data
cond_no_train = np.array([cond_no_train_list[i] for i in degree])

fig, ax = plt.subplots(figsize=(10, 6))
ax.semilogy(degree, cond_no_train, color='black', linestyle='-', linewidth=3, marker='o', markersize=8)
ax.xaxis.set_major_locator(MultipleLocator(1))
plt.ylim(10**(0), 10**(7))
plt.xlabel(r'$M$')
plt.ylabel(r'$\kappa$')
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig('cond_no_poly.pdf', dpi=1080)
plt.show()
# ===== Class competition =====

#residual computation
y_predict = vandermonde(col1, degree[2]).to_numpy() @ np.array(x_list[degree[2]])
#print(y_predict.shape)
residue = col2.to_numpy().flatten() - y_predict        
print(f'Residual: ', residue)
print((col2.to_numpy().flatten()).shape)
#print(residue.shape)
plt.scatter(col2.to_numpy().flatten(), residue)
plt.ylim(-0.02, 0.02)

#closed form regularization (Ridge regression)
lam = 1e-7 #regularization parameter
x_reg = (np.linalg.inv(vandermonde(col1, degree[2]).T @ vandermonde(col1, degree[2]) + lam * np.eye(vandermonde(col1, degree[2]).shape[1])) @ vandermonde(col1, degree[2]).T @ col2).to_numpy().flatten() #$\omega = (A^TA + \lambda I)^{-1}A^T\vec{y}$
x_reg[np.abs(x_reg) < 1e-6] = 0 #threshold for co-efficient
print(x_reg)
eq_reg = linear_reg(x_reg) #convert co-efficient to polynomial equation
print(eq_reg)
mse_reg = np.mean((abs(col2.to_numpy().flatten() - vandermonde(col1, degree[2]).to_numpy() @ x_reg))**2) #MSE
print(f'Regularized MSE: ', mse_reg)

#plotting noisy data and estimated actual polynomial
fig, ax = plt.subplots(figsize=(10, 6))
plt.scatter(col1, col2, color='red', label='Data')
eq_plot = plot(eq_reg.rhs, (symbols("x"), col1.to_numpy().flatten().min(), col1.to_numpy().flatten().max()), show=False)

eq_plot[0].line_color = 'black'
eq_plot[0].line_width = 3
eq_plot[0].label = 'Polynomial fit'

for line in eq_plot:
    ax.plot(*line.get_points(), color=line.line_color,
            linewidth=line.line_width, linestyle='-', 
            label=line.label)
plt.xlabel('$t$')
plt.ylabel('$y$')
plt.xlim(col1.to_numpy().flatten().min(), col1.to_numpy().flatten().max())
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'poly_fit.pdf', dpi=1080)
plt.show()

"""
x_reg_list = {}
lse_reg_list = {}
mse_reg_list = {}
degree_ = np.linspace(1, 9, 9, dtype=int)
for i in degree_:
    vandermonde_ = vandermonde(col1, i)
    x = (np.linalg.inv(vandermonde_.T @ vandermonde_) @ vandermonde_.T @ col2).to_numpy().flatten()
    x[np.abs(x) < 1e-6] = 0
    y = col2.to_numpy().flatten()
    lse = np.sum((abs(y - vandermonde_ @ x))**2)
    mse = np.mean((abs(y - vandermonde_ @ x))**2)
    x_reg_list[i] = x
    lse_reg_list[i] = lse
    mse_reg_list[i] = mse

print(x_reg_list)
print('Co-efficients:', x_reg_list)
print('LSE:', lse_reg_list)
print('MSE:', mse_reg_list)

reg = RANSACRegressor(estimator=LinearRegression(fit_intercept=False), residual_threshold=1, min_samples=0.5)
reg.fit(vandermonde(col1, degree[2]).to_numpy(), col2.to_numpy().flatten())
x_reg = reg.estimator_.coef_
print(x_reg.shape)
mse_reg = np.mean((abs(col2.to_numpy().flatten() - (vandermonde(col1, degree[2]).to_numpy() @ x_reg)))**2)
print(mse_reg)
"""