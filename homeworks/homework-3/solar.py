import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# read csv
data = np.loadtxt("SolarData-Hw3.csv", delimiter=",")
t = data[:,0]
print(t)
theta = data[:,1]

# prior for mean and variance
mu_0 = 0.0
tau_0_sq = 49

# maximum likelihood and noise variance
omega_mle = np.sum(t*theta)/np.sum(t*t)
print("MLE = ", omega_mle)
var = 1/(len(t)-1)*np.sum((theta-omega_mle*t)**2)

# constants
G = 6.6743e-11
r = 1.56479e11

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

print("Bayesian posterior mean = ", mu[-1])
print("Bayesian posterior variance = ", tau_sq[-1])
print("Bayesian posterior mean/std. deviation = ", mu[-1]/np.sqrt(tau_sq[-1]))

# parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['CMU Serif']
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 20
mpl.rcParams['axes.unicode_minus'] = False

# plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(np.arange(1, len(t)+1), mu*1e7, "-o", color="black", label=r'$\mu_k$')
ax.axhline(omega_mle*1e7, color="red", linestyle='--', linewidth=2, label=r'$\hat{\omega}_{MLE}$')
ax.fill_between(np.arange(1, len(t)+1), (mu-1.96*np.sqrt(tau_sq))*1e7, (mu+1.96*np.sqrt(tau_sq))*1e7,
                 color="blue", alpha=0.25, rasterized=True, label="95% credible interval")
plt.xlabel(r"$k$")
plt.ylabel(r"$\omega \times 10^7$ $(rad/s)$")
plt.legend(loc="upper right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.xlim(0, 13)
plt.ylim(-1, 5)
plt.savefig(f'omega_bayes.pdf', dpi=1080)
plt.show()

# Monte Carlo sampling
np.random.seed(32) # for reproducibility
n = 1000000 # no. of realizations
omega_smpl = np.random.normal(mu[-1], np.sqrt(tau_sq[-1]), n)
mass_smpl = (r**3 / G) * (omega_smpl**2)

# mean and bounds
mass_mean = np.mean(mass_smpl)
mass_var = np.var(mass_smpl, ddof=1)
#mass_low, mass_high = np.percentile(mass_smpl, [2.5, 97.5])

print(f"Posterior mean mass: {mass_mean}")
print(f"95% credible interval: [{mass_mean} - {1.96*np.sqrt(mass_var)}, {mass_mean} + {1.96*np.sqrt(mass_var)}]")
#print(f"95% credible interval: [{mass_mean} - {mass_mean - mass_low}, {mass_mean} + {-mass_mean + mass_high}] kg")

# plot of sampless mass
fig, ax = plt.subplots(figsize=(15, 6))

ax.hist(mass_smpl/1e30, bins=100, color='green', edgecolor='black', alpha=0.5, density=True, label='Posterior samples')

# 95% credible interval
ax.axvline(mass_mean/1e30, color='black', lw=3, label='Posterior mean')
ax.axvline((mass_mean-1.96*np.sqrt(mass_var))/1e30, color='blue', lw=3, linestyle='--', label='2.5% quantile')
ax.axvline((mass_mean+1.96*np.sqrt(mass_var))/1e30, color='blue', lw=3, linestyle='-.', label='97.5% quantile')

# errorbar
ax.errorbar(mass_mean/1e30, 0.002, xerr=[[1.96*np.sqrt(mass_var)/1e30], 
            [1.96*np.sqrt(mass_var)/1e30]], fmt='o', color='red', capsize=6, 
            lw=3, label='95% credible interval')

ax.set_xlabel(r"Mass of star (in $10^{30}kg$)")
ax.set_ylabel(r"$Density$")
ax.legend(frameon=False)
plt.savefig(f'hist.pdf', dpi=1080)
plt.show()