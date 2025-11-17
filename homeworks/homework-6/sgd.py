import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
import jax.numpy as jnp
import optax
import time

# ===== C. Dataset and setup =====

def generate_data(scale_noise, N, seed=None):
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, N)
    s = 1 + t + t*t
    n = rng.normal(loc=0.0, scale=scale_noise, size=N)
    y = s + n
    return t, y, s, n
    
t_all, y_all, s_all, n_all       = generate_data(0.0, 1000, seed=0)
t_train, t_test, y_train, y_test = train_test_split(t_all,y_all,test_size=0.2,random_state=42)

# parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 20
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True

fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(t_train, y_train, color="blue", label="Training data")
ax.scatter(t_test, y_test, color="red", label="Testing data")
plt.xlabel(r"$t$")
plt.ylabel(r"$y$")
plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"gen_data.pdf", dpi=1080)
plt.show()

# ===== D. Write and test the model and loss functions =====

print("\n===== D. Write and test the model and loss functions =====")

# function for Vandermonde matrix A
def vandermonde(t, M):
    t = jnp.asarray(t)
    powers = jnp.arange(M + 1)
    return t[:, None] ** powers[None, :]


# 1. function for model $A\vec{\omega}$
def model(w, A):
    return A @ w

# 2. function for loss function ($\lambda = 0$)
def loss(w, A, y):
    y_pred = model(w, A)
    r = y_pred - y
    return jnp.dot(r, r)
    
# 3. function for MSE
def mse(w, A, y):
    return loss(w, A, y) / len(y)

# sanity test
print("\nSANITY TEST")

# noise-free data generation
N = 1000
t, y, s, _ = generate_data(scale_noise=0.0, N=N, seed=1)

# degree of polynomial and Vandermonde construction
M = 2 
A = vandermonde(t, M)

# True weights for polynomial
w_true = jnp.array([1.0, 1.0, 1.0])

# wrong weights to check for high loss
w_bad = jnp.array([1.0, 0.5, 0.5])

# compute true losses and MSE
loss_true = loss(w_true, A, y)
mse_true = mse(w_true, A, y)

# compute losses and MSE for wrong weights
loss_bad = loss(w_bad, A, y)
mse_bad = mse(w_bad, A, y)

print("\nTrue weights: w_true =", w_true)
print("Loss(w_true) =", float(loss_true))
print("MSE(w_true)  =", float(mse_true))

print("\nBad weights: w_bad =", w_bad)
print("Loss(w_bad)  =", float(loss_bad))
print("MSE(w_bad)   =", float(mse_bad))

# train and test MSE 
A_train = vandermonde(t_train, M)
A_test  = vandermonde(t_test,  M)
train_mse_true = mse(w_true, A_train, y_train)
test_mse_true  = mse(w_true, A_test,  y_test)

print("\nTraining MSE (true weights) =", float(train_mse_true))
print("Testing  MSE (true weights) =", float(test_mse_true))

# ====== E. SGD optimization ======

print("\n====== E. SGD optimization ======")

# 1. Full batch SGD

M = 3
lam = 0
alpha = 0.2       
epochs = 100000   
B = len(t_train) 

# construct train and test Vandermonde matrices
A_train = vandermonde(t_train, M)
A_test  = vandermonde(t_test,  M)
y_train_jnp = jnp.asarray(y_train)
y_test_jnp  = jnp.asarray(y_test)

# closed form solution
AtA = A_train.T @ A_train
Aty = A_train.T @ y_train_jnp
w_opt = jnp.linalg.solve(AtA + lam*jnp.eye(M+1), Aty)

print("\nClosed-form: omega_opt = ", w_opt)

def vandermonde_scaled(t, M):
    t = jnp.asarray(t)
    t = 2*(t - 0.5)        # scale to [-1, 1]
    powers = jnp.arange(M + 1)
    return t[:, None] ** powers[None, :]
    
# compute Hessian $H = 2A^TA$
hessian = 2 * (A_train.T @ A_train) 

# compute the eigenvalues of the Hessian
eigen_hessian = np.linalg.eigvals(hessian)

# maximum eigenvalue of the Hessian
max_eigen_hessian = np.max(eigen_hessian)

# learning rate upper bound
alpha_max = 2.0 / max_eigen_hessian

print("\nMaximum eigenvalue of Hessian and upper bound of learning rate:")
print(f"\nlambda_max(H) = {max_eigen_hessian:.3e}")
print(f"alpha_max = 2 / lambda_max(H) = {alpha_max:.3e}")

# SGD diverges or converges?
if alpha > alpha_max:
    print(f"\nSGD should theoretically diverge as alpha = {alpha} is greater than alpha_max = {alpha_max:.3e}")
    print(f"\nLet's try with scaled Vandermonde...")
    
    # scaled Vandermonde
    A_train = vandermonde_scaled(t_train, M)
    A_test  = vandermonde_scaled(t_test,  M)  
    
    # compute Hessian $H = 2A^TA$
    hessian = 2 * (A_train.T @ A_train) 
    
    # compute the eigenvalues of the Hessian
    eigen_hessian = np.linalg.eigvals(hessian)
    
    # maximum eigenvalue of the Hessian
    max_eigen_hessian = np.max(eigen_hessian)
    
    # learning rate upper bound
    alpha_max = 2.0 / max_eigen_hessian
    
    print("\nAfter scaling...")
    print(f"\nlambda_max(H) = {max_eigen_hessian:.3e}")
    print(f"alpha_max = 2 / lambda_max(H) = {alpha_max:.3e}")
    if alpha > alpha_max:
        print(f"\nSGD should theoretically still diverge.")
    else:
        print(f"\nSGD should theoretically converge")
else:
    print(f"\nSGD should theoretically converge as alpha = {alpha} is less than or equal to alpha_max = {alpha_max:.3e}.")

# initialization of weights
key = jax.random.PRNGKey(0)
w = jax.random.normal(key, (M+1,))

# Optax SGD with gradient clipping
#optimizer = optax.sgd(learning_rate=alpha)

optimizer = optax.chain(
    optax.clip(0.5),  # Lower clipping threshold for better gradient stability
    optax.sgd(learning_rate=alpha)
)

opt_state = optimizer.init(w)

# update step
@jax.jit
def step(w, opt_state, A, y):
    grads = jax.grad(loss)(w, A, y)       
    updates, opt_state = optimizer.update(grads, opt_state)
    w = optax.apply_updates(w, updates)
    return w, opt_state

# initialize train, test, distance arrays
train_mse_hist, test_mse_hist, dist_hist = [], [], []

# full batch training loop
for epoch in range(epochs):
    w, opt_state = step(w, opt_state, A_train, y_train_jnp)

    # compute real losses
    train_mse_hist.append(mse(w, A_train, y_train_jnp))
    test_mse_hist.append(mse(w, A_test, y_test_jnp))
    dist_hist.append(jnp.linalg.norm(w - w_opt))

print("\nFinal SGD weights:", w)
print("Distance to closed-form =", float(dist_hist[-1]))

# parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 20
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True

# plot for training & testing MSE vs. epoch
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(train_mse_hist, "b-", lw=2, label="Train MSE")
ax.loglog(test_mse_hist, "r--", lw=2, label="Test MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(loc="upper right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"mse_wclip.pdf", dpi=1080)
plt.show()

# plot L2 norm of error vs. epoch
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(dist_hist, "k-", lw=2)
plt.xlabel("Epoch")
plt.ylabel(r"$\| \omega_i - \omega_{\text{opt}} \|_2$")
#plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"l2_wclip.pdf", dpi=1080)
plt.show()

# now try with scaled data

# scaled Vandermonde 
A_train_s = vandermonde_scaled(t_train, M)
A_test_s  = vandermonde_scaled(t_test,  M)

# closed form solution in scaled basis
AtA_s = A_train_s.T @ A_train_s
Aty_s = A_train_s.T @ y_train_jnp
w_opt_s = jnp.linalg.solve(AtA_s + lam*jnp.eye(M+1), Aty_s)

print("\nClosed-form solution (scaled basis):", w_opt_s)

# new random weights
key2 = jax.random.PRNGKey(123)
w2 = jax.random.normal(key2, (M+1,))

# optimizer with clipping
optimizer2 = optax.chain(
    optax.clip(0.5),  # Lower clipping threshold for better gradient stability
    optax.sgd(learning_rate=alpha)
)
opt_state2 = optimizer2.init(w2)

# new step function for scaled run
@jax.jit
def step2(w, opt_state, A, y):
    grads = jax.grad(loss)(w, A, y)
    updates, opt_state = optimizer2.update(grads, opt_state)
    w = optax.apply_updates(w, updates)
    return w, opt_state

# histories
train_mse_hist_2 = []
test_mse_hist_2  = []
dist_hist_2      = []

# training loop
for epoch in range(epochs):
    w2, opt_state2 = step2(w2, opt_state2, A_train_s, y_train_jnp)

    train_mse_hist_2.append(mse(w2, A_train_s, y_train_jnp))
    test_mse_hist_2.append(mse(w2, A_test_s, y_test_jnp))
    dist_hist_2.append(jnp.linalg.norm(w2 - w_opt_s))

print("\nFinal SGD weights (scaled):", w2)
print("Distance to scaled closed-form:", float(dist_hist_2[-1]))

# plot scaled training & testing MSE vs. epoch
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(train_mse_hist_2, "b-", lw=2, label="Train MSE (scaled)")
ax.loglog(test_mse_hist_2, "r--", lw=2, label="Test MSE (scaled)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(loc="upper right", frameon=False)
plt.savefig("mse_scaled.pdf", dpi=1080)
plt.show()

# plot scaled L2 norm of error vs. epoch
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(dist_hist_2, "k-", lw=2)
plt.xlabel("Epoch")
plt.ylabel(r"$\| w_i - w_{\text{opt,scaled}} \|_2$")
plt.savefig("l2_scaled.pdf", dpi=1080)
plt.show()

# 2. Mini-batch SGD

# unscaled mini-batch
epochs_mb = 2000
batch_size = 32 
num_batches = len(t_train)

# shuffle indices each epoch
def get_batches(A, y, batch_size):
    N = len(y)
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield A[idx], y[idx]

# fresh weights
key = jax.random.PRNGKey(420)
w_mb = jax.random.normal(key, (M+1,))

# optimizer
optimizer_mb = optax.chain(
    optax.clip(0.5),  # Lower clipping threshold for better gradient stability
    optax.sgd(learning_rate=alpha)
)
opt_state_mb = optimizer_mb.init(w_mb)

# mini-batch training step
@jax.jit
def mb_step(w, opt_state, A_batch, y_batch):
    grads = jax.grad(loss)(w, A_batch, y_batch)
    updates, opt_state = optimizer_mb.update(grads, opt_state)
    w = optax.apply_updates(w, updates)
    return w, opt_state

# history arrays
train_mse_hist_mb = []
test_mse_hist_mb  = []
dist_hist_mb      = []

# mini-batch SGD loop
for epoch in range(epochs_mb):
    # iterate over randomized batches
    for A_b, y_b in get_batches(A_train, y_train_jnp, batch_size):
        w_mb, opt_state_mb = mb_step(w_mb, opt_state_mb, A_b, y_b)
    # record metrics each epoch
    train_mse_hist_mb.append(mse(w_mb, A_train, y_train_jnp))
    test_mse_hist_mb.append(mse(w_mb, A_test,  y_test_jnp))
    dist_hist_mb.append(jnp.linalg.norm(w_mb - w_opt))
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Train MSE = {train_mse_hist_mb[-1]:.6f}, Dist = {dist_hist_mb[-1]:.3e}")

print("\nFinal mini-batch SGD weights:", w_mb)
print("Distance to closed-form =", float(dist_hist_mb[-1]))

# plots
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(train_mse_hist_mb, "b-", lw=2, label="Train MSE (mini-batch)")
ax.loglog(test_mse_hist_mb,  "r--", lw=2, label="Test MSE (mini-batch)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(loc="upper right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("mse_mb_unscaled.pdf", dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(dist_hist_mb, "k-", lw=2)
plt.xlabel("Epoch")
plt.ylabel(r"$\| w - w_{\text{opt}} \|_2$")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("l2_mb_unscaled.pdf", dpi=1080)
plt.show()

# scaled mini-batch
epochs_mb_s = 2000
batch_size_s = 32

# build scaled Vandermonde
A_train_s = vandermonde_scaled(t_train, M)
A_test_s  = vandermonde_scaled(t_test,  M)

# closed form solution in scaled basis
AtA_s = A_train_s.T @ A_train_s
Aty_s = A_train_s.T @ y_train_jnp
w_opt_s = jnp.linalg.solve(AtA_s + lam*jnp.eye(M+1), Aty_s)

print("\nClosed-form solution (scaled basis) for mini-batch:", w_opt_s)

# fresh weights for scaled mini-batch
key_s = jax.random.PRNGKey(777)
w_mb_s = jax.random.normal(key_s, (M+1,))

# optimizer (same learning rate and clipping)
optimizer_mb_s = optax.chain(
    optax.clip(0.5),
    optax.sgd(learning_rate=alpha)
)
opt_state_mb_s = optimizer_mb_s.init(w_mb_s)

# batching function for scaled data
def get_batches_s(A, y, batch_size):
    N = len(y)
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield A[idx], y[idx]

# JIT-compiled update for scaled mini-batch
@jax.jit
def mb_step_s(w, opt_state, A_batch, y_batch):
    grads = jax.grad(loss)(w, A_batch, y_batch)
    updates, opt_state = optimizer_mb_s.update(grads, opt_state)
    w = optax.apply_updates(w, updates)
    return w, opt_state

# histories
train_mse_hist_mb_s = []
test_mse_hist_mb_s  = []
dist_hist_mb_s      = []

# training loop
for epoch in range(epochs_mb_s):
    for A_b, y_b in get_batches_s(A_train_s, y_train_jnp, batch_size_s):
        w_mb_s, opt_state_mb_s = mb_step_s(w_mb_s, opt_state_mb_s, A_b, y_b)

    # record per-epoch progress
    train_mse_hist_mb_s.append(mse(w_mb_s, A_train_s, y_train_jnp))
    test_mse_hist_mb_s.append(mse(w_mb_s, A_test_s, y_test_jnp))
    dist_hist_mb_s.append(jnp.linalg.norm(w_mb_s - w_opt_s))

    if epoch % 100 == 0:
        print(f"[SCALED] Epoch {epoch}: Train MSE = {train_mse_hist_mb_s[-1]:.6f}, Dist = {dist_hist_mb_s[-1]:.3e}")

print("\nFinal mini-batch SGD weights (scaled):", w_mb_s)
print("Distance to scaled closed-form =", float(dist_hist_mb_s[-1]))

# plots
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(train_mse_hist_mb_s, "b-", lw=2, label="Train MSE (scaled mini-batch)")
ax.loglog(test_mse_hist_mb_s,  "r--", lw=2, label="Test MSE (scaled mini-batch)")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(loc="upper right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("mse_mb_scaled.pdf", dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(dist_hist_mb_s, "k-", lw=2)
plt.xlabel("Epoch")
plt.ylabel(r"$\| w - w_{\text{opt,scaled}} \|_2$")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("l2_mb_scaled.pdf", dpi=1080)
plt.show()

# F. Batch size study

print("\n===== F. Batch size study =====")

# parameters
batch_sizes = [1, 16, 64, len(t_train)]
epochs_F = 100000  # maximum epoch count
alpha_F = 0.1
M = 3
target_mse = 1e-4  # target

results = {}

# early stopping parameters
patience = 5  # no. of epochs with no significant improvement before stopping
min_delta = 1e-5  # minimum change in MSE to be considered as improvement

# define the mini-batch step function
@jax.jit
def mb_step_F(w, opt_state, A_batch, y_batch):
    grads = jax.grad(loss)(w, A_batch, y_batch)
    updates, opt_state = optimizer_F.update(grads, opt_state)
    w = optax.apply_updates(w, updates)
    return w, opt_state

# define the batch creation function
def get_batches_F(A, y, batch_size):
    N = len(y)
    perm = np.random.permutation(N)
    for i in range(0, N, batch_size):
        idx = perm[i:i+batch_size]
        yield A[idx], y[idx]

# pre-calculated scaled data
A_train_scaled = A_train_s 
A_test_scaled = A_test_s   

for B in batch_sizes:
    print(f"\n--- Running SGD for batch size B = {B} ---")

    # fresh weights
    key_F = jax.random.PRNGKey(999 + B)
    w_F = jax.random.normal(key_F, (M + 1,))
    
    # optimizer with gradient clipping
    optimizer_F = optax.chain(
        optax.clip(0.5),  # clip gradients to prevent explosion
        optax.sgd(learning_rate=alpha_F)
    )
    opt_state_F = optimizer_F.init(w_F)

    # histories
    train_hist, test_hist = [], []
    t0 = time.time()
    reached_time = None
    previous_mse = float('inf')  # initialize previous MSE as a large value

    # First, ensure training runs for at least 100 epochs
    for epoch in range(100):  # first mandatory 100 epochs
        # process batches using scaled data
        for A_b, y_b in get_batches_F(A_train_scaled, y_train_jnp, B):
            w_F, opt_state_F = mb_step_F(w_F, opt_state_F, A_b, y_b)

        # record metrics
        train_mse_val = mse(w_F, A_train_scaled, y_train_jnp)
        test_mse_val  = mse(w_F, A_test_scaled, y_test_jnp)
        train_hist.append(train_mse_val)
        test_hist.append(test_mse_val)

        # time to target test MSE
        if reached_time is None and float(test_mse_val) < target_mse:
            reached_time = time.time() - t0

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: test MSE = {float(test_mse_val):.3e}")

    # after the first 100 epochs, check for target MSE and early stopping
    for epoch in range(100, epochs_F):  
        # process batches using scaled data
        for A_b, y_b in get_batches_F(A_train_scaled, y_train_jnp, B):
            w_F, opt_state_F = mb_step_F(w_F, opt_state_F, A_b, y_b)

        # record metrics
        train_mse_val = mse(w_F, A_train_scaled, y_train_jnp)
        test_mse_val  = mse(w_F, A_test_scaled, y_test_jnp)
        train_hist.append(train_mse_val)
        test_hist.append(test_mse_val)

        # time to target test MSE
        if reached_time is None and float(test_mse_val) < target_mse:
            reached_time = time.time() - t0

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: test MSE = {float(test_mse_val):.3e}")

        # Check if target MSE is reached
        if test_mse_val <= target_mse:
            print(f"Target MSE reached at epoch {epoch}. Stopping training.")
            break

        # Early stopping criterion after running for 100 epochs
        if abs(previous_mse - test_mse_val) < min_delta:
            patience -= 1
            if patience == 0:
                print(f"Early stopping at epoch {epoch} due to lack of improvement.")
                break
        else:
            patience = 5  # reset patience if there was significant improvement

        previous_mse = test_mse_val  # update previous MSE for the next epoch

    total_time = time.time() - t0

    results[B] = {
        "train_hist": train_hist,
        "test_hist":  test_hist,
        "time_to_target": reached_time,
        "final_test_mse": float(test_hist[-1]),
        "total_time": total_time
    }

    print(f"Final test MSE = {results[B]['final_test_mse']:.3e}")
    print(f"Time to reach target: {results[B]['time_to_target']}")
    print(f"Total time = {total_time:.2f} sec")

# plot train and test MSE

for B in batch_sizes:
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.loglog(results[B]["train_hist"], lw=2, color='blue', label=f"Train B={B}")
    ax.loglog(results[B]["test_hist"], lw=2, linestyle="--", color='red', label=f"Test B={B}")

    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f"batch_{B}_mse.pdf", dpi=1080)
    plt.show()

# ===== G. Learning-rate study =====

print("\n===== G. Learning-rate study =====")

# parameters
b = 32
epoch_max = 5000       
learning_rates = [0.1, 0.01, 0.001]  # list of learning rates
target_mse = 1e-6
#min_delta = 1e-4
#patience_init = 5
#min_epochs = 1000

# reuse scaled vandermonde 
a_train_scaled = A_train_s
a_test_scaled  = A_test_s

# closed-form solution in scaled basis
ata_s = a_train_scaled.T @ a_train_scaled
aty_s = a_train_scaled.T @ y_train_jnp
w_opt_s = jnp.linalg.solve(ata_s + lam*jnp.eye(M+1), aty_s)

# function to create mini-batches
def get_batches_g():
    n = len(y_train_jnp)
    perm = np.random.permutation(n)
    for i in range(0, n, b):
        idx = perm[i:i+b]
        yield a_train_scaled[idx], y_train_jnp[idx]

# general training loop for any optimizer
def run_optimizer(opt, label, alpha0):
    key = jax.random.PRNGKey(999 + hash(label) % 100)
    w = jax.random.normal(key, (M+1,))
    opt_state = opt.init(w)

    train_hist = []
    test_hist  = []
    dist_hist  = []

    reached_time = None
    t0 = time.time()

    @jax.jit
    def step(w, opt_state, A_b, y_b):
        grads = jax.grad(loss)(w, A_b, y_b)
        updates, opt_state = opt.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return w, opt_state

    for epoch in range(epoch_max):     

        # update using mini-batches
        for A_b, y_b in get_batches_g():
            w, opt_state = step(w, opt_state, A_b, y_b)

        # compute metrics
        train_mse = float(mse(w, a_train_scaled, y_train_jnp))
        test_mse  = float(mse(w, a_test_scaled,  y_test_jnp))
        dist_w    = float(jnp.linalg.norm(w - w_opt_s))

        train_hist.append(train_mse)
        test_hist.append(test_mse)
        dist_hist.append(dist_w)

        # record time to reach target mse
        if reached_time is None and test_mse < target_mse:
            reached_time = time.time() - t0
        
        # periodic debug output
        if epoch % 1000 == 0:
            print(f"{label}: epoch {epoch}, test mse = {test_mse:.3e}")

    print(f"{label}: final mse = {train_mse:.3e}, dist = {dist_w:.3e}")
    return train_hist, test_hist, dist_hist

# define optimizers
def get_optimizer(alpha0):
    opt_const = optax.chain(
        optax.clip(0.5),
        optax.sgd(learning_rate=alpha0)
    )

    opt_momentum = optax.chain(
        optax.clip(0.5),
        optax.trace(decay=0.9, nesterov=True),
        optax.scale(-alpha0)
    )

    schedule_cos = optax.cosine_decay_schedule(
        init_value=alpha0,
        decay_steps=epoch_max        
    )
    opt_cosine = optax.chain(
        optax.clip(0.5),
        optax.sgd(learning_rate=schedule_cos)
    )
    
    return opt_const, opt_momentum, opt_cosine

# loop over all learning rates
for lr in learning_rates:
    print(f"\nRunning with learning rate = {lr}")
    # define the optimizers for the current learning rate
    opt_const, opt_momentum, opt_cosine = get_optimizer(lr)

    # run all optimizers
    train_const, test_const, dist_const = run_optimizer(opt_const,   "constant lr", lr)
    train_mom,   test_mom,   dist_mom   = run_optimizer(opt_momentum,"momentum beta=0.9", lr)
    train_cos,   test_cos,   dist_cos   = run_optimizer(opt_cosine,  "cosine decay", lr)

    # plot train mse vs epoch
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.loglog(train_const, color="blue", lw=2, label="Constant lr")
    ax.loglog(train_mom,   color="red", lw=2, label=rf"Momentum $\beta=0.9$")
    ax.loglog(train_cos,   color="green", lw=2, label="Cosine decay")

    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f"optimizer_comp_mse_lr{lr}.pdf", dpi=1080)
    plt.show()

    # plot test mse vs epoch
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.loglog(test_const, color="blue", lw=2, label="Constant lr")
    ax.loglog(test_mom,   color="red", lw=2, label=rf"Momentum $\beta=0.9$")
    ax.loglog(test_cos,   color="green", lw=2, label="Cosine decay")

    plt.xlabel("Epoch")
    plt.ylabel("Test MSE")
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f"optimizer_comp_test_mse_lr{lr}.pdf", dpi=1080)
    plt.show()

    # plot l2 norm vs epoch
    fig, ax = plt.subplots(figsize=(15, 6))

    ax.loglog(dist_const, color="blue", lw=2, label="constant lr")
    ax.loglog(dist_mom,   color="red", lw=2, label=rf"momentum $\beta=0.9$")
    ax.loglog(dist_cos,   color="green", lw=2, label="cosine decay")

    plt.xlabel("Epoch")
    plt.ylabel(r"$\| w - w_{\text{opt,scaled}} \|_2$")
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f"optimizer_comp_lr{lr}.pdf", dpi=1080)
    plt.show()
    
# ===== H. SGD vs another optimizer =====

print("\n===== H. SGD vs another optimizer =====")

# parameters for this comparison
b = 32
epoch_max = 5000       
learning_rates = [0.1, 0.01, 0.001]  # list of learning rates
target_mse = 1e-6

# optimizers (momentum, adam)
def get_optimizer_momentum(alpha0):
    return optax.chain(
        optax.clip(0.5),
        optax.trace(decay=0.9, nesterov=True),
        optax.scale(-alpha0)
    )

def get_optimizer_adam(alpha0):
    return optax.chain(
        optax.clip(0.5),
        optax.adam(learning_rate=alpha0)
    )

# function to create mini-batches
def get_batches_g(A, y, batch_size):
    n = len(y)
    perm = np.random.permutation(n)
    for i in range(0, n, batch_size):
        idx = perm[i:i+batch_size]
        yield A[idx], y[idx]
        
# general training loop for any optimizer
def run_optimizer(opt, label, alpha0, A_train_scaled, A_test_scaled, y_train_jnp, y_test_jnp):
    key = jax.random.PRNGKey(999 + hash(label) % 100)
    w = jax.random.normal(key, (M+1,))
    opt_state = opt.init(w)

    train_hist = []
    test_hist  = []
    dist_hist  = []

    reached_time = None
    t0 = time.time()

    @jax.jit
    def step(w, opt_state, A_b, y_b):
        grads = jax.grad(loss)(w, A_b, y_b)
        updates, opt_state = opt.update(grads, opt_state, w)
        w = optax.apply_updates(w, updates)
        return w, opt_state

    for epoch in range(epoch_max):     
        # update using mini-batches
        for A_b, y_b in get_batches_g(A_train_scaled, y_train_jnp, b):
            w, opt_state = step(w, opt_state, A_b, y_b)

        # compute metrics
        train_mse = float(mse(w, A_train_scaled, y_train_jnp))
        test_mse  = float(mse(w, A_test_scaled,  y_test_jnp))
        dist_w    = float(jnp.linalg.norm(w - w_opt_s))

        train_hist.append(train_mse)
        test_hist.append(test_mse)
        dist_hist.append(dist_w)

        # record time to reach target mse
        if reached_time is None and test_mse < target_mse:
            reached_time = time.time() - t0
        
        # periodic debug output
        if epoch % 1000 == 0:
            print(f"{label}: epoch {epoch}, test mse = {test_mse:.3e}")

    print(f"{label}: final mse = {train_mse:.3e}, dist = {dist_w:.3e}")
    return train_hist, test_hist, dist_hist

# loop over all learning rates for comparison of Momentum and Adam
for lr in learning_rates:
    print(f"\nRunning with learning rate = {lr}")
    
    # get optimizers for Momentum and Adam
    opt_momentum = get_optimizer_momentum(lr)
    opt_adam = get_optimizer_adam(lr)

    # run
    train_mom, test_mom, dist_mom = run_optimizer(opt_momentum, "Momentum beta=0.9", lr, A_train_s, A_test_s, y_train_jnp, y_test_jnp)
    train_adam, test_adam, dist_adam = run_optimizer(opt_adam, "Adam optimizer", lr, A_train_s, A_test_s, y_train_jnp, y_test_jnp)

    # plot train mse vs epoch
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.loglog(train_mom, color="red", lw=2, label=rf"Momentum $\beta=0.9$")
    ax.loglog(train_adam, color="purple", lw=2, label="Adam")
    plt.xlabel("Epoch")
    plt.ylabel("Train MSE")
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f"train_optimizer_comp_mse_lr{lr}_momentum_adam.pdf", dpi=1080)
    plt.show()

    # Ppot test mse vs epoch
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.loglog(test_mom, color="red", lw=2, label=rf"Momentum $\beta=0.9$")
    ax.loglog(test_adam, color="purple", lw=2, label="Adam")
    plt.xlabel("Epoch")
    plt.ylabel("Test MSE")
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f"test_optimizer_comp_mse_lr{lr}_momentum_adam.pdf", dpi=1080)
    plt.show()

    # plot l2 norm vs epoch
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.loglog(dist_mom, color="red", lw=2, label=rf"Momentum $\beta=0.9$")
    ax.loglog(dist_adam, color="purple", lw=2, label="Adam")
    plt.xlabel("Epoch")
    plt.ylabel(r"$\| w - w_{\text{opt,scaled}} \|_2$")
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f"norm_optimizer_comp_lr{lr}_momentum_adam.pdf", dpi=1080)
    plt.show()