import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# read csv
data = np.loadtxt("hw3.csv", delimiter=",")
t = data[:,0]
theta = data[:,1]

# ===== Problem-1 =====

# compute maximum likelihood solution
omega_mle = np.sum(t*theta)/np.sum(t*t)
#print(np.sum(t*theta))
#print(np.sum(t*t))
print("1(c) Maximum likelihood solution: ", omega_mle)

# parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 24
mpl.rcParams['axes.unicode_minus'] = False

# plot
fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(t, theta, color='red', label=r"$\theta$", linewidth = 3)
ax.plot(t, omega_mle*t, linestyle='-', color='black', label=r"$\hat\omega_{MLE} t$", linewidth = 3)

plt.xlim(0, 1250)
ax.set_xticks(np.arange(0, 1250, 300))
plt.ylim(0, 14)
plt.xlabel(r"$t$ ($s$)")
plt.ylabel("$Angular$ $coordinate$ ($rad$)")
plt.legend(loc='upper left', frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'1d.pdf', dpi=1080)
plt.show()

# compute noise variance
var = 1/(len(t)-1)*np.sum((theta-omega_mle*t)**2)
print("1(e) Noise variance: ", var)

# ===== Problem-2 =====

# selected priors for mean and variance
mu_0 = 0
tau_0_sq = 49

# compute posterior variance
tau_n_sq = 1/((1/tau_0_sq)+((1/var)*np.sum(t**2)))
print("2(b) Bayesian posterior variance: ", tau_n_sq)

# compute posterior mean
mu_n = tau_n_sq*((mu_0/tau_0_sq) + (1/var)*np.sum(t*theta))
print("2(b) Bayesian posterior mean: ", mu_n)

# ===== Problem-3 =====

# initialize posterior mean, variance, $\sum t^2$, $sum t\theta$
mu = np.zeros(len(t))
tau_sq = np.zeros(len(t))
s_tt = 0.0
s_ttheta = 0.0

# sequential update of posterior mean and variance
for k in range(len(t)):
    s_tt += t[k]**2
    s_ttheta += t[k]*theta[k]
    tau_sq_ = 1/((1/tau_0_sq)+((1/var)*s_tt))
    mu_ = tau_sq_*((mu_0/tau_0_sq) + (1/var)*s_ttheta)
    tau_sq[k] = tau_sq_
    mu[k] = mu_

print("3(a) mu_k: ", mu)
print("3(a) tau_sq_k: ", tau_sq)

# plot
fig, ax = plt.subplots(figsize=(15, 6))
plt.plot(np.arange(1, len(t)+1), mu, "-o", color="black", label=r'$\mu_k$')
plt.axhline(omega_mle, color="red", linestyle='--', linewidth=2, label=r'$\hat{\omega}_{MLE}$')
plt.fill_between(np.arange(1, len(t)+1), mu-1.96*np.sqrt(tau_sq), mu+1.96*np.sqrt(tau_sq),
                 color="blue", alpha=0.25, rasterized=True, label="95% credible interval")
plt.xlabel(r"$k$")
plt.ylabel(r"$\omega$ $(rad/s)$")
plt.legend(loc="lower right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.ylim(-0.005, 0.02)
plt.savefig(f'3b-1.pdf', dpi=1080)
plt.show()

# zoomed plot
fig, ax = plt.subplots(figsize=(15, 6))
plt.plot(np.arange(1, len(t)+1), mu, "-o", color="black", label=r'$\mu_k$')
plt.axhline(omega_mle, color="red", ls='--', linewidth=2, label=r'$\hat{\omega}_{MLE}$')
plt.fill_between(np.arange(1, len(t)+1), mu-1.96*np.sqrt(tau_sq), mu+1.96*np.sqrt(tau_sq),
                 color="blue", alpha=0.25, rasterized=True, label="95% credible interval")
plt.xlabel(r"$k$")
plt.ylabel(r"$\omega$ $(rad/s)$")
plt.legend(loc="upper right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.ylim(0.01, 0.011)
plt.savefig(f'3b-2.pdf', dpi=1080)
plt.show()

# ===== Problem-4 =====

# set prior for variance an entropy
tau_0_sq_ = 1
h_0 = 1/2+1/2*np.log(2*np.pi*tau_0_sq_)

# reinitialize variance, $\sum t^2$; initialize entropy and difference in entropy
tau_sq = np.zeros(len(t))
h = np.zeros(len(t))
dh = np.zeros(len(t))
s_tt = 0.0

# sequential update of entropy and difference in entropy
for k in range(len(t)):
    s_tt += t[k]**2
    tau_sq_ = 1/((1/tau_0_sq_)+((1/var)*s_tt))
    h_ = 1/2+1/2*np.log(2*np.pi*tau_sq_)
    if k == 0:
        dh_ = h_ - h_0
    else:
        dh_ = h_ - h[k-1]
    tau_sq[k] = tau_sq_
    h[k] = h_
    dh[k] = dh_

print("4(a) tau_sq_k: ", tau_sq)
print("4(a) H_k: ", h)
print("4(c) dH_k: ", dh)

# plot
fig, ax = plt.subplots(figsize=(15, 6))

plt.plot(np.arange(1, len(t)+1), h, "-o", linewidth=2, color="black")

plt.xlabel(r"$k$")
plt.ylabel(r"$H_k$")
plt.xlim(-1, 101)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'4b.pdf', dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))

plt.plot(np.arange(1, len(t)+1), dh, "-o", linewidth=2, color="black")

plt.xlabel(r"$k$")
plt.ylabel(r"$\Delta H_k$")
plt.xlim(0, 101)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'4c.pdf', dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))

plt.plot(np.arange(1, len(t)+1), tau_sq, "-o", linewidth=2, color="black")

plt.xlabel(r"$k$")
plt.ylabel(r"$\tau_k^2$")
plt.xlim(0, 101)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'4d.pdf', dpi=1080)
plt.show()