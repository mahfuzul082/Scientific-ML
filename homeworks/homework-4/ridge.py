import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import Ridge

def generate_data(scale_noise, N, seed=56):
    """ Generate a noisy time series dataset {t_i, y_i}_{i=1}^N.
    Parameters
    ----------
    A : float
    Amplitude of the signal (A >= 0)
    N : int
    Number of data points. Time points uniformly sampled from [0, 2pi]
    seed : int or None, optional
    Random seed for reproducibility """
    
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, N)
    dt = (2*np.pi) / (N - 1)
    A=np.sqrt(np.pi/dt)
    s_hat = np.sqrt(dt / np.pi) * np.sin(2*np.pi*t)
    # ||s_hat||_2 == 1 exactly on this grid
    s = A * s_hat
    n = rng.normal(loc=0.0, scale=scale_noise, size=N)
    y = s + n
    return t, y, s, n, s_hat, dt

# ===== 1. Ridge implementation, diagnostics, and interpretation =====

# 1. Implementation and comparison
# generate testing and training data
x_data,y_data,s,n,s_hat, dt = generate_data(.1,100)
t_train = x_data[::10]
y_train = y_data[::10]
s_train = s[::10]
plt.plot(t_train,y_train,'bo')
plt.plot(t_train,s_train,'r--')
plt.plot(x_data,s,'k--')
plt.show()
print("1. Implementation and comparison")
print("\nTraining data (t, y):\n", np.column_stack((t_train, y_train)))

# generate testing data
# TODO for YOU: Write code to remove the training data below
# t_test and y_test will have 90 data points

t_test  = np.delete(x_data, np.arange(0, len(x_data), 10))
y_test  = np.delete(y_data, np.arange(0, len(y_data), 10))
print("\nTesting data (t, y):\n", np.column_stack((t_test, y_test)))

# function to construct Vandermonde matrix
def vandermonde(t, M):
    return np.vstack([t**i for i in range(M+1)]).T

# function to construct $\gamma$ matrix 
def gamma(M):
    g = np.ones(M+1)
    g[0] = 0.0
    return np.diag(g)

M = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]) # polynomial orders
lambda_ = np.array([0, 1e-6, 1e-4, 1e-2, 1e0, 1e2]) # regularization factors

omega_train_list = {} # omega for training data
mse_train_list = {} # training MSE
mse_test_list = {} # testing MSE
cond_no_train_list = {} # training condition no. w/o regularization factor
cond_no_ridge_train_list = {} # training condition no. w/ regularization factor
cond_no_test_list = {} # testing condition no. w/o regularization factor
cond_no_ridge_test_list = {} # testing condition no. w/ regularization factor

for i in M:
    A_train = vandermonde(t_train, i)
    A_test = vandermonde(t_test, i)
    for j in lambda_:
        omega_train = np.linalg.solve(((A_train.T @ A_train) + 
                                    j * (gamma(i).T @ gamma(i))), (A_train.T @ y_train))
        omega_train_list[(i, j)] = omega_train
        y_train_predict = A_train @ omega_train
        y_test_predict = A_test @ omega_train
        mse_train = np.mean((abs(y_train_predict - y_train))**2)
        mse_train_list[(i, j)] = mse_train # update traing MSE list
        mse_test = np.mean((abs(y_test_predict - y_test))**2)
        mse_test_list[(i, j)] = mse_test # update testing MSE list
        # compute and update condition no. lists
        cond_no_train = np.linalg.cond(A_train.T @ A_train, 2) 
        cond_no_train_list[(i, j)] = cond_no_train
        cond_no_ridge_train = np.linalg.cond((A_train.T @ A_train + j * gamma(i).T @ gamma(i)), 2) 
        cond_no_ridge_train_list[(i, j)] = cond_no_ridge_train
        cond_no_test = np.linalg.cond(A_test.T @ A_test, 2) 
        cond_no_test_list[(i, j)] = cond_no_test
        cond_no_ridge_test = np.linalg.cond((A_test.T @ A_test + j * gamma(i).T @ gamma(i)), 2) 
        cond_no_ridge_test_list[(i, j)] = cond_no_ridge_test
        
print("\n$omega$:", omega_train_list)
print("\nMSE (training):", mse_train_list)
print("\nMSE (testing):", mse_test_list)
print("\nCondition no. w/o regularization (training):", cond_no_train_list)
print("\nCondition no. w/ regularization (training):", cond_no_ridge_train_list)
print("\nCondition no. w/o regularization (testing):", cond_no_test_list)
print("\nCondition no. w/ regularization (testing):", cond_no_ridge_test_list)

df = pd.DataFrame([[M_val, lam, mse_train_list[(M_val, lam)], 
                    mse_test_list[(M_val, lam)], 
                    cond_no_train_list[(M_val, lam)],
                    cond_no_ridge_train_list[(M_val, lam)], 
                    cond_no_test_list[(M_val, lam)], cond_no_ridge_test_list[(M_val, lam)]]
    for M_val in M for lam in lambda_],
columns=["M", "$\lambda$", "MSE_train", "MSE_test", "$kappa(A^TA)_train$", 
         "$kappa(A^TA+\lambda\gamma^T\gamma)_train$", "$kappa(A^TA)_test$", 
         "$kappa(A^TA+\lambda\gamma^T\gamma)_test$"])
print("\nTable for all combinations:\n", df)

# parameters for plotting
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["CMU Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 20
mpl.rcParams["axes.unicode_minus"] = False

# mse plots
fig, ax = plt.subplots(figsize=(10, 10))
markers = ["o", "s", "^"]
linestyles = ["-", "--"] 

for i, M_val in enumerate(M):
    subset = df[df["M"] == M_val]
    ax.loglog(subset["$\\lambda$"], subset["MSE_test"], 
              marker=markers[i % len(markers)], markersize=10,
              linestyle=linestyles[i % len(linestyles)], linewidth=3,
              label=fr"$M={M_val}$")

plt.xlabel(r"$\lambda$")
plt.ylabel(r"Test $MSE$")
plt.ylim(1e-2, 1e0)
plt.legend(frameon=False, loc="best", ncol=3)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("ridge_mse.pdf", dpi=1080)
plt.show()

# condition no. plot w/o regularization
fig, ax = plt.subplots(figsize=(10, 10))

subset = df[df["$\\lambda$"] == 0].sort_values("M")

ax.semilogy(subset["M"], subset["$kappa(A^TA)_train$"], "-o", markersize=10, linewidth=3, color="blue", label="Training")
ax.semilogy(subset["M"], subset["$kappa(A^TA)_test$"], "--o", markersize=10, linewidth=3, color="red", label="Testing")

plt.xlabel(r"$M$")
plt.ylabel(r"$\kappa(A^TA)$")
plt.ylim(1e1, 1e15)
plt.legend(frameon=False, loc="upper left")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("cond.pdf", dpi=1080)
plt.show()

# condition no. plot w/ regularizations
fig, ax = plt.subplots(figsize=(10, 10))
markers = ["o", "s", "^"]
linestyles = ["-", "--"] 

for i, M_val in enumerate(M):
    subset = df[df["M"] == M_val]
    ax.loglog(subset["$\\lambda$"], subset["$kappa(A^TA+\\lambda\\gamma^T\\gamma)_train$"],
              marker=markers[i % len(markers)], markersize=10, 
              linestyle=linestyles[i % len(linestyles)], linewidth=3,
              label=fr"$M={M_val}$")

plt.xlabel(r"$\lambda$")
plt.ylabel(r"Train $\kappa(A^TA+\lambda\Gamma^T\Gamma)$")
plt.ylim(1e1, 1e8)
plt.legend(frameon=False, loc="upper right", ncol=3)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("train_ridge_cond.pdf", dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
markers = ["o", "s", "^"]
linestyles = ["-", "--"] 

for i, M_val in enumerate(M):
    subset = df[df["M"] == M_val]
    ax.loglog(subset["$\\lambda$"], subset["$kappa(A^TA+\\lambda\\gamma^T\\gamma)_test$"],
              marker=markers[i % len(markers)], markersize=10, 
              linestyle=linestyles[i % len(linestyles)], linewidth=3,
              label=fr"$M={M_val}$")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"Test $\kappa(A^TA+\lambda\Gamma^T\Gamma)$")
plt.ylim(1e0, 1e9)
plt.legend(frameon=False, loc='upper right', ncol=3)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("test_ridge_cond.pdf", dpi=1080)
plt.show()

# 2. Equivalence check with scikit-learn
diff = []

print("\n2. Equivalence check with scikit-learn")
for M_val in M:
    A_train = vandermonde(t_train, M_val)
    I = np.eye(M_val + 1)
    for lam in lambda_:
        # closed form solution
        omega_closed = np.linalg.solve((A_train.T @ A_train) + 
                                       lam * (I.T @ I), A_train.T @ y_train)

        # scikit-learn solution
        ridge = Ridge(alpha=lam, fit_intercept=False, solver='auto')
        ridge.fit(A_train, y_train)
        omega_sklearn = ridge.coef_

        # compute l2 norm
        diff_ = np.linalg.norm(omega_closed - omega_sklearn)
        diff.append([M_val, lam, diff_])

        print(f"\nM={M_val:2d}, lambda={lam:8.1e},  error={diff_:.2e}")

# dataframe
df_diff = pd.DataFrame(diff, columns=["M", "$lambda$", "l2"])
print("\nCoefficient L2-norm differences across all (M, $\lambda$):\n")
print(df_diff)

# ===== 2. Model selection over polynomial degree and regularization =====

# 1. Grid search over (M, $\lambda$) 
M_hat, lambda_hat = min(mse_test_list, key=mse_test_list.get)
min_mse_test = mse_test_list[M_hat, lambda_hat]

print(f"\n1. Grid search over (M, $\lambda$)")
print(f"\nBest (M_hat, Lambda) = ({M_hat}, {lambda_hat})")
print(f"\nMinimum Test MSE = {min_mse_test}")

# 2. Final reflections
cases = [
    ("Best", M_hat, lambda_hat),
    ("Ill-conditioned", 7, 0),
    ("Over-regularized", 7, 1e0)
]

coeffs_list = {}

print(f"\n2. Final reflections")

# pick the ($M, \lambda$) for comparison
for label, M_val, lam in cases:
    coeffs_list[label] = omega_train_list[(M_val, lam)]
    print(f"\n{label} coefficients (M={M_val}, $\lambda$={lam:g}):")
    print(coeffs_list[label])

# noise-free true signal
_, _, s_true, _, _, _ = generate_data(0.0, 100, seed=56)

# plot for comparison
fig, ax = plt.subplots(figsize=(15, 6))

ax.plot(x_data, s_true, "k-", linewidth=3, label="True signal")

colors = ["red", "blue", "green"]
linestyles = ["-", "--", "-."]

for (label, M_val, lam), color, ls in zip(cases, colors, linestyles):
    A_plot = vandermonde(x_data, M_val)
    y_fit = A_plot @ coeffs_list[label]
    ax.plot(x_data, y_fit, color=color, linestyle=ls, linewidth=3,
            label=rf"{label} ({M_val}, {lam:g})")

plt.xlabel(r"$t$")
plt.ylabel("$y$")
plt.legend(frameon=False, loc="upper right")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("signal_fit.pdf", dpi=1080)
plt.show()