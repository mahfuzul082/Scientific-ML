import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

def generate_data(A, N, seed=None):
    """
    Generate a noisy time series dataset {t_i, y_i}_{i=1}^N.
    Parameters
    ----------
    A : float
    Amplitude of the signal (A >= 0)
    N : int
    Number of data points. Time points uniformly sampled from [0, 2pi]
    seed : int or None, optional
    Random seed for reproducibility 
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2*np.pi, N)
    dt = (2*np.pi) / (N - 1)
    s_hat = np.sqrt(dt / np.pi) * np.sin(t) # ||s_hat||_2 == 1 exactly on this grid
    s = A * s_hat
    n = rng.normal(loc=0.0, scale=1.0, size=N)
    y = s + n
    return t, y, s, n, s_hat, dt

# ===== Emperical and theoretical Gaussian distribution =====
    
A = [0, 3, 10] # input amplitudes
N = 1000 # no. of data points
seed = 32 # for reproducibility
n_realization = 50000 # no. of realizations

rho_list = {} # $\rho(A)$ list

for i in A:
    rho_arr = [] # $\rho(A)$ array for realizations
    for j in range(n_realization):
        t, y, s, n, s_hat, dt = generate_data(i, N, seed + j)
        rho = np.dot(y, s_hat)
        rho_arr.append(rho) # update new realization
    rho_list[i] = np.array(rho_arr) # update list

for i in A:
    print(f"A={i}: rho_mean={rho_list[i].mean():.8f}, rho_var={rho_list[i].var():.8f}")

# parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 23
mpl.rcParams['axes.unicode_minus'] = False

# individual plots of distributions for different $A$ values
x = np.linspace(-5, 15, 1000)
clr = ['red', 'green', 'blue']
lbl = [r"$A=0$", r"$A=3$", r"$A=10$"]

for i, color in zip(A, clr):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # empirical histogram
    ax.hist(rho_list[i], bins=30, density=True, alpha=0.5, 
            label=r"$Empirical$", color = color, edgecolor='black')
    
    # theoretical gaussian line plot
    ax.plot(x, norm.pdf(x, loc=i, scale=1), color=color, ls = '-', lw=4,
            label=r"$Theoretical$")
    
    plt.xlim(i-6, i+6)
    ax.set_xticks(np.arange(i-6, i+6, 2))
    plt.xlabel(fr"$\rho$")
    plt.ylabel(fr"$Density$")
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f'hist{i}.pdf', dpi=1080)
    plt.show()

# single plot of the three distributions
fig, ax = plt.subplots(figsize=(12, 6))

for i, color, label in zip(A, clr, lbl):
    # empirical histogram
    ax.hist(rho_list[i], bins=30, alpha=0.5, density=True,
             color=color, label=label, edgecolor='black')

    # theoretical gaussian line plot
    ax.plot(x, norm.pdf(x, loc=i, scale=1), linestyle='-', color=color, linewidth = 4)

plt.xlim(-5, 15)
ax.set_xticks([-5, -3, 0, 3, 5, 7, 10, 12, 15])
plt.xlabel(r'$\rho(A)$')
plt.ylabel('$Density$')
plt.legend(loc='upper left', bbox_to_anchor=(0.45, 0.99), frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'hist.pdf', dpi=1080)
plt.show()

# ===== Construction of confusion matrix =====

N_c = 1000 # no. of data points
seed = 32 # for reproducibility
n_dataset = 1000 # no. of mock datasets

rng = np.random.default_rng(seed)
signal_label = rng.choice([0, 3], size=n_dataset) # randomly assignment of $A=0$ or $A=3$

rho_arr = [] # $rho$ array

# generating each mock dataset with amplitude
for i, A in enumerate(signal_label):
    t, y, s, n, s_hat, dt = generate_data(A, N, seed + i)
    rho_arr.append(np.dot(y, s_hat)) # update $rho$ array
    
rho_ = np.array(rho_arr) 

actual = (signal_label == 3).astype(int) # 1 if signal, 0 if no signal

thres = np.linspace(-3, 8, 2000) # 2000 threshold points evaluated
min_diff = 10 # initialized difference between false negative and false positive

for rho0 in thres:
    predict = (rho_ >= rho0).astype(int) # 1 if the condition is true, 0 if false
    
    # condiions of four states of confusion matrix
    false_negative = np.sum((predict==0) & (actual==1))
    true_negative = np.sum((predict==0) & (actual==0))
    true_positive = np.sum((predict==1) & (actual==1))
    false_positive = np.sum((predict==1) & (actual==0))
    
    # difference between false negative and false positive
    diff = abs(false_negative - false_positive)
    
    # minimizing the difference
    if diff < min_diff:
        min_diff = diff
        best_thres = rho0
        best_confus = [[true_negative, false_positive], [false_negative, true_positive]]
        
print(fr'Best threshold = {best_thres:.6f}')
print('Confusion matrix: ')
print(np.array(best_confus))

# plots
rho0 = rho_[signal_label == 0]
rho3 = rho_[signal_label == 3]

fig, ax = plt.subplots(figsize=(12, 6))

# histograms
ax.hist(rho0, bins=50, density=True, alpha=0.5, color='red',
         edgecolor='black', label=r'$A=0$ (no signal)')
ax.hist(rho3, bins=50, density=True, alpha=0.5, color='green',
         edgecolor='black', label=r'$A=3$ (signal)')

# theoretical gaussian line plot
ax.plot(x, norm.pdf(x, loc=0, scale=1), linestyle='-', color='red', linewidth = 4)
ax.plot(x, norm.pdf(x, loc=3, scale=1), linestyle='-', color='green', linewidth = 4)

# threshold line
plt.axvline(best_thres, color='k', linestyle='--', linewidth=4,
            label=fr'$\rho_0$ = {best_thres:.6f}')

plt.xlim(-6, 9)
ax.set_xticks(np.arange(-6, 9, 3))
plt.xlabel(r'$\rho(A)$')
plt.ylabel('$Density$')
plt.legend(loc='upper left', frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'confusion.pdf', dpi=1080)
plt.show()

# ===== Class competition =====

# read csv
data = np.loadtxt("hw2.csv", delimiter=",")
t = data[:,0]
y = data[:,1]
print('Length of y:', len(y))

# scaling function
def scaling(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) # mean absolute deviation to determine the sparsity of data
    sigma = 1.4826 * mad # standart deviation
    
    # centering and rescaling (if possible)
    if sigma > 0:
        return (x - med) / sigma
    else:
        return x - med

# loop over different size of bins
for N in range(50, 1001, 50):
    dt = t[1] - t[0] # timestep
    t_ = t[:N]
    omega = 2*np.pi / (t_[-1] - t_[0])
    s_hat = np.cos(omega * (t_ - t_[0]))  # discrete template as cosine function

    rho_g = []
    segments = []  # for optional template refinement
    
    # $rho$ computation
    for i in range(0, len(y), N):
        y_ = y[i:i+N]
        if len(y_) < N:
            continue
        y_scaled = scaling(y_)
        rho = np.dot(y_scaled, s_hat)
        rho_g.append(rho)
        segments.append((i, y_)) 

    rho_g = np.array(rho_g)

    # computing noise mean, spread and threshold by taking 60% center data
    rho_lo, rho_hi = 20, 80
    lo, hi = np.percentile(rho_g, [rho_lo, rho_hi])
    noise_chunk = rho_g[(rho_g >= lo) & (rho_g <= hi)]
    mu_ = np.median(noise_chunk)
    mad = np.median(np.abs(noise_chunk - mu_))
    if mad > 0:
        sigma_ = 1.4826 * mad
    else:
        sigma_ = np.std(noise_chunk, ddof=1)

    alpha = 0.01
    rho0 = mu_ + sigma_ * norm.ppf(1 - alpha)

    detect = np.where(np.abs(rho_g) >= rho0)[0] # condition for detection
    sig_count = len(detect) # counting signals
    print(f'N={N}: Estimated number of signals = {sig_count} / {len(rho_g)}')

    # refine template from detected bins
    refined_template = s_hat  # fallback default
    if len(detect) > 0:
        detected_segs = []
        for idx in detect:
            i, y_seg = segments[idx]
            if len(y_seg) == N:
                y_seg = y[i:i+N]
                a = np.dot(y_seg, s_hat) / np.dot(s_hat, s_hat)
                signal_estimate = a * s_hat
                detected_segs.append(signal_estimate)
        refined_template = np.mean(detected_segs, axis=0)

    # histogram for each N
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(rho_g, bins=max(10, min(40, len(rho_g)//2)), density=True,
            alpha=0.5, color='blue', edgecolor='black', label=r"$\rho$")
    x = np.linspace(min(rho_g)-0.5, max(rho_g)+0.5, 500)
    ax.plot(x, norm.pdf(x, loc=mu_, scale=sigma_), 'b-', lw=2,
            label=fr"$\mathcal{{N}}({mu_:.3f},{sigma_**2:.3f})$")
    ax.axvline(rho0, color='black', ls='--', lw=3,
               label=fr'$\rho_0={rho0:.3f}$')
    plt.xlabel(r'$\rho$')
    plt.ylabel(r'$Density$')
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f'hist_sig{N}.pdf', dpi = 1080)
    plt.show()
    
    # signal line plot for each N
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(t, y, linewidth=1, label="Noisy data")

    for i, j in enumerate(detect):
        seg_start = j * N
        seg_t = t[seg_start:seg_start+N]
        if len(seg_t) < N:
            continue
    
        y_seg = y[seg_start:seg_start+N]
        a = np.dot(y_seg, refined_template) / np.dot(refined_template, refined_template)
        ax.plot(seg_t, a * refined_template, 'r--', lw=2,
                label='Signal' if i == 0 else "")

    plt.xlabel('$t$')
    plt.ylabel('$A$')
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f'signal{N}.pdf', dpi = 1080)
    plt.show()