import numpy as np
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt
import matplotlib as mpl
from numpy.linalg import solve
from sklearn.mixture import GaussianMixture
from scipy.integrate import simpson

# Load and shuffle data
data = np.loadtxt("hw4_train.csv", delimiter=",")
t = data[:, 0]
y = data[:, 1]

# normalize time to [-1, 1]
t_norm = 2 * (t - np.min(t)) / (np.max(t) - np.min(t)) - 1

# sort by time
id = np.argsort(t_norm)
t_norm = t_norm[id]
y = y[id]

# Use early time as training, later as validation
split = int(0.7 * len(t_norm))
t_train, y_train = t_norm[:split], y[:split]
t_test,  y_test  = t_norm[split:], y[split:]

# ===== Ridge regression model =====

# construct design matrix
def design_matrix(t, n_poly=2, omega=None):
    t = np.asarray(t, dtype=float)
    if omega is None:
        omega = [1.0, 2.0, 3.0]

    cols = []

    # orthogonal polynomial trend (Legendre basis)
    trend = legvander(t, n_poly)
    for i in range(trend.shape[1]):
        cols.append(trend[:, i])

    # harmonic oscillations
    for w in omega:
        cols.append(np.cos(w * np.pi * t))
        cols.append(np.sin(w * np.pi * t))

    return np.column_stack(cols)

# train and test design matrix
A_train = design_matrix(t_train, n_poly=2, omega=[1.0, 2.0])
A_test  = design_matrix(t_test,  n_poly=2, omega=[1.0, 2.0])

print("\nCondition no. of $A^TA$ (train):", np.linalg.cond(A_train.T @ A_train))
print("\nCondition no. of $A^TA$ (test):", np.linalg.cond(A_test.T @ A_test))


# regularization function
def gamma_t_gamma(n_poly, omega):

    weights = []
    # polynomial terms lightly penalized
    for i in range(n_poly + 1):
        if i == 0:
            weights.append(0.0)
        else:
            weights.append(1e-3)
    # harmonics penalized proportionally to the square of frequency
    for k, w in enumerate(omega, start=1):
        strength = (k**2)
        weights += [strength, strength]
    return np.diag(weights)

# ridge regression function
def ridge_regression(A, y, Gamma_t_Gamma, lam=1e-3, trim=0.02):
    # initial fit
    omega = solve(A.T @ A + lam * Gamma_t_Gamma, A.T @ y)
    y_pred = A @ omega
    resid = y - y_pred

    # trim extreme residuals (slamming)
    if trim > 0:
        threshold = np.quantile(np.abs(resid), 1 - trim)
        mask = np.abs(resid) < threshold
        A_trim, y_trim = A[mask], y[mask]
        omega = solve(A_trim.T @ A_trim + lam * Gamma_t_Gamma, A_trim.T @ y_trim)
    
    return omega

Gamma_t_Gamma = gamma_t_gamma(n_poly=2, omega=[1.0, 2.0])

# grid search for best fit
lambda_ = np.logspace(-6, 2, 40)
mse_train, mse_test = [], []
cond_train, cond_test = [], []

for lam in lambda_:
    theta = ridge_regression(A_train, y_train, Gamma_t_Gamma, lam=lam, trim=0.02)
    y_train_pred = A_train @ theta
    y_test_pred  = A_test  @ theta
    mse_train.append(np.mean((y_train_pred - y_train)**2))
    mse_test.append(np.mean((y_test_pred - y_test)**2))
    
    # regularized condition no.
    cond_train.append(np.linalg.cond(A_train.T @ A_train + lam * Gamma_t_Gamma))
    cond_test.append(np.linalg.cond(A_test.T  @ A_test  + lam * Gamma_t_Gamma))

best_id = np.argmin(mse_test)
lambda_best = lambda_[best_id]
print(f"\nBest $\lambda$ = {lambda_best:.4e}, Test MSE = {mse_test[best_id]:.8f}")

# parameters for plotting
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["CMU Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
mpl.rcParams["axes.unicode_minus"] = False

# condition number plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(lambda_, cond_train, "bo-", ms=10, lw=3, label=fr"Train $\kappa(A^TA+\lambda\Gamma^T\Gamma)$")
ax.loglog(lambda_, cond_test,  "ro-", ms=10, lw=3, label=fr"Test $\kappa(A^TA+\lambda\Gamma^T\Gamma)$")
ax.axvline(lambda_best, color="k", lw=3, ls="--", label=fr"Best $\lambda$={lambda_best:.4f}")
plt.xlabel(r"$\lambda$")
plt.ylabel(r"$\kappa(A^TA+\lambda\Gamma^T\Gamma)$")
plt.ylim(1e0, 1e11)
ax.legend(frameon=False, loc='best')
ax.tick_params(axis="both", which="both", direction="in")
plt.savefig("marine_cond.pdf", dpi=1080)
plt.show()

# mse plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(lambda_, mse_train, "bo-", ms=10, lw= 3, label="Train $MSE$")
ax.loglog(lambda_, mse_test, "ro-", ms=10, lw=3, label=r"Test $MSE$")
ax.axvline(lambda_best, color="k", lw=3, ls="--", label=rf"Best $\lambda$={lambda_best:.4f}")
plt.xlabel("$\lambda$")
plt.ylabel("$MSE$")
plt.legend(frameon=False, loc="best")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("marine_mse.pdf", dpi=1080)
plt.show()

# best fit model
theta_best = ridge_regression(A_train, y_train, Gamma_t_Gamma, lam=lambda_best, trim=0.02)
print("\nBest $\theta$ =", theta_best)

y_pred_train = A_train @ theta_best
y_pred_test  = A_test  @ theta_best
print("\nPredicted acceleration (train): \n", y_pred_train)
print("\nPredicted acceleration (test): \n", y_pred_test)

# de-normalize time
t_o = 0.5 * (t_norm + 1) * (np.max(t) - np.min(t)) + np.min(t)
t_train_o = t_o[:split]
t_test_o  = t_o[split:]

# prediction plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t_train_o, y_train, "bo", alpha=0.5, label="Training data")
ax.plot(t_test_o,  y_test,  "ro", alpha=0.5, label="Testing data")
ax.plot(t_train_o, y_pred_train, color="orange", ls="-", lw=3, label="Fit (train)")
ax.plot(t_test_o,  y_pred_test,  "k-", lw=3, label='Fit (test)')
plt.xlabel("$t$")
plt.ylabel("$y$")
plt.ylim(-3, 4)
plt.legend(frameon=False, loc="best", ncol=2)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("marine_predict.pdf", dpi=1080)
plt.show()

# ===== GMM noise model =====

# residuals
resid_train = y_train - y_pred_train
resid_test  = y_test  - y_pred_test

# residual plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.hist(resid_train, bins=60, density=True, color="blue", edgecolor="black", alpha=0.6)
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$Density$")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("marine_residual.pdf", dpi=1080)
plt.show()

# 1D GMM with 2 components
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=23)
gmm.fit(resid_train.reshape(-1, 1))

means   = gmm.means_.ravel()
vars_   = gmm.covariances_.ravel()
stds    = np.sqrt(gmm.covariances_).ravel()
weights = gmm.weights_.ravel()

print("\nGMM parameters using training residual")
for i, (w, m, v, s) in enumerate(zip(weights, means, vars_, stds)):
    print(f"\nComponent {i+1}: weight={w:.4f}, mean={m:.4f}, variance={v:.4f}, std={s:.4f}")

# GMM plot
x_grid = np.linspace(min(resid_train), max(resid_train), 600).reshape(-1, 1)
pdf_total = np.exp(gmm.score_samples(x_grid))

# PDF of individal component
pdf_components = np.array([
    w * (1 / (np.sqrt(2*np.pi) * s)) * np.exp(-0.5 * ((x_grid - m) / s)**2)
    for w, m, s in zip(weights, means, stds)
])

colors = ["red", "orange"]
labels = ["BG sea state", "Slamming"]
fig, ax = plt.subplots(figsize=(15, 6))
ax.hist(resid_train, bins=60, density=True, color="blue", edgecolor="black", alpha=0.6, label="$\epsilon$")
ax.plot(x_grid, pdf_total, "k-", lw=3, label="Fitted GMM")
for k, (pdf, c, lbl) in enumerate(zip(pdf_components, colors, labels)):
    ax.plot(x_grid, pdf, "--", lw=3, color=c, label=f"Component {k+1} ({lbl})")
plt.xlabel(r"$\epsilon$")
plt.ylabel(r"$Density$")
plt.tick_params(axis="both", which="both", direction="in")
plt.legend(frameon=False, loc="best")
plt.savefig("marine_gmm.pdf", dpi=1080)
plt.show()

# ===== Challenge Submission I =====

# acceleration^4
a4 = np.abs(y_pred_test)**4

# Trapezoidal rule
vdv_trap = (np.trapz(a4, t_test_o)) ** 0.25

# Simpson's rule
vdv_simp = (simpson(a4, t_test_o)) ** 0.25

print("\nVibration Dose Value (VDV):")
print(f"\nTrapezoidal rule: {vdv_trap:.8f}")
print(f"\nSimpsons 1/3rd rule : {vdv_simp:.8f}")

# ===== Challenge Submission II =====

id_bgc = np.argmin(stds)
sigma_bgc_pred = stds[id_bgc]
id_slm = np.argmax(stds)

sigma_ratio = stds[id_slm] / stds[id_bgc]
p_slam = weights[id_slm]

print("\nGMM Noise:")
print(f"\nBackground sea state standard deviation = {sigma_bgc_pred:.8f}")
print(f"\nSlamming standard deviation = {stds[id_slm]:.8f}")
print(f"\nRatio of standard deviation of slamming and background sea = {sigma_ratio:.8f}")
print(f"\nSlamming probability = {p_slam:.4f}")