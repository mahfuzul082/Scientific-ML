import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
import jax.numpy as jnp
import optax

# ===== B. Data and target function =====

# target function f*
def f_star(t):
    return np.exp(-3.0 * t) * np.sin(8 * np.pi * t)

# create train and test sets
rng = np.random.default_rng(0)
t_train = rng.uniform(0.0, 1.0, size=128)
y_train = f_star(t_train)

t_test = np.linspace(0.0, 1.0, 2048, endpoint=True)
y_test = f_star(t_test)

# plotting params
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 20
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True

# plot the target function on a dense grid
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t_test, y_test, 'k-', lw=2)
plt.xlabel(r"$t$")
plt.ylabel(r"$f^*(t)$")
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("fstar.pdf", dpi=1080)
plt.show()

# ===== C. Two-layer network model =====

# relu activation
def relu(x):
    return jnp.maximum(0.0, x)

# tanh activation
def tanh(x):
    return jnp.tanh(x)

# model initialization
def init_model(rng, m):
    key_w, key_t, key_a, key_c = jax.random.split(rng, 4)

    # wide input weights
    w = 30.0 * jax.random.normal(key_w, (m,))

    # kink locations uniformly across [0,1]
    t0 = jax.random.uniform(key_t, (m,), minval=0.0, maxval=1.0)

    # bias
    b = -w * t0

    # small output layer weights
    a = 0.1 * jax.random.normal(key_a, (m,))

    c = jnp.array(0.0) # output bias

    return {"w": w, "b": b, "a": a, "c": c}

# forward pass
def model_apply(params, t, activation):
    # compute $w_j t + b_j$ for all hidden units
    z = params["w"] * t[:, None] + params["b"]

    # apply activation
    h = activation(z)

    # output
    y_pred = jnp.dot(h, params["a"]) + params["c"]
    return y_pred

# mse loss function
def mse_loss(params, t, y, activation):
    y_pred = model_apply(params, t, activation)
    r = y_pred - y
    return jnp.mean(r * r)

# ===== D. Implementation and visualization =====

# width
m = 16

# number of epochs
epochs_d = 250000

# optimizer (adam)
optimizer_d = optax.adam(learning_rate=1e-3)

# initialize parameters
rng = jax.random.PRNGKey(299)
params_d = init_model(rng, m)

opt_state_d = optimizer_d.init(params_d)

# choose activation outside the jit
activation = relu

# jit-compiled update step
@jax.jit
def step(params, opt_state, t, y):
    loss_val, grads = jax.value_and_grad(mse_loss)(params, t, y, activation)
    updates, opt_state = optimizer_d.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

# training history
train_hist_d = []
test_hist_d  = []

# convert training data to jax arrays
t_train_j = jnp.asarray(t_train)
y_train_j = jnp.asarray(y_train)
t_test_j  = jnp.asarray(t_test)
y_test_j  = jnp.asarray(y_test)

# training loop
for epoch in range(epochs_d):
    # shuffle each epoch
    key_epoch = jax.random.PRNGKey(epoch)
    perm = jax.random.permutation(key_epoch, len(t_train_j))
    t_batch = t_train_j[perm]
    y_batch = y_train_j[perm]

    params_d, opt_state_d, train_loss = step(params_d, opt_state_d,
                                             t_batch, y_batch)
    
    train_hist_d.append(float(train_loss))
    
    # compute test mse
    test_loss = mse_loss(params_d, t_test_j, y_test_j, activation)
    test_hist_d.append(float(test_loss))

    if epoch % 500 == 0:
        print(f"epoch {epoch}, train mse = {train_loss:.4e}")

print(f"\nfinal train mse = {train_hist_d[-1]:.4e}")
print(f"final test  mse = {test_hist_d[-1]:.4e}")

# plot mse vs. epoch
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(train_hist_d, "b-", lw=2, label="Train MSE")
ax.loglog(test_hist_d,  "r--", lw=2, label="Test MSE")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("mse_m16.pdf", dpi=1080)
plt.show()

# plot model and true function
y_pred_d = model_apply(params_d, t_test_j, activation)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t_test, y_test,     "k-", lw=2, label=r"$f^*(t)$")
ax.plot(t_test, y_pred_d,   "r--", lw=2, label=r"$\tilde f(t)$, m=16")
plt.xlabel(r"$t$")
plt.ylabel(r"$f(t)$")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("compare_m16.pdf", dpi=1080)
plt.show()

# ===== E. Empirical convergence with width =====

# params
width_list = [2, 4, 8, 16, 32, 64, 128]
epochs_e = 20000
restarts = 3

E2_vals = []
Einf_vals = []

# optimizer (adam)
optimizer_e = optax.adam(learning_rate=1e-3)

@jax.jit
def step_e(params, opt_state, t, y):
    loss_val, grads = jax.value_and_grad(mse_loss)(params, t, y, relu)
    updates, opt_state = optimizer_e.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

for m in width_list:
    print(f"\n=== width m = {m} ===")

    # store errors from restarts
    err2_list = []
    errinf_list = []

    for r in range(restarts):
        print(f" restart {r}")

        # init params
        key = jax.random.PRNGKey(1000 + 17*m + r)
        params = init_model(key, m)
        opt_state = optimizer_e.init(params)

        # train
        for epoch in range(epochs_e):
            key_epoch = jax.random.PRNGKey(epoch)
            perm = jax.random.permutation(key_epoch, len(t_train_j))
            t_batch = t_train_j[perm]
            y_batch = y_train_j[perm]
            params, opt_state, train_loss = step_e(params, opt_state, t_batch, y_batch)

        # compute test prediction
        y_pred = model_apply(params, t_test_j, relu)

        # compute errors (convert to numpy first)
        residual = np.array(y_pred) - np.array(y_test)
        E2   = np.sqrt(np.mean(residual**2))
        Einf = np.max(np.abs(residual))

        err2_list.append(E2)
        errinf_list.append(Einf)

    # take median across restarts
    E2_vals.append(np.median(err2_list))
    Einf_vals.append(np.median(errinf_list))

# convert to numpy arrays
width_arr = np.array(width_list)
E2_vals   = np.array(E2_vals)
Einf_vals = np.array(Einf_vals)

# log-log slope estimation
logm   = np.log(width_arr)
logE2  = np.log(E2_vals)
logEin = np.log(Einf_vals)

alpha_E2,   _ = np.polyfit(logm, logE2, 1)
alpha_Einf, _ = np.polyfit(logm, logEin, 1)

print("\nestimated slopes (ReLU):")
print(f"  alpha_E2   ~ {-alpha_E2:.3f}")
print(f"  alpha_Einf ~ {-alpha_Einf:.3f}")

# plot E2(m)
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(width_arr, E2_vals, "b-o", lw=2, label=r"$E_2(m)$")
plt.xlabel(rf"Width, $m$")
plt.ylabel(r"$E_2(m)$")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("E2_vs_m.pdf", dpi=1080)
plt.show()

# plot of Einf(m)
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(width_arr, Einf_vals, "r--s", lw=2, label=r"$E_\infty(m)$")
plt.xlabel(rf"Width, $m$")
plt.ylabel(r"$E_\infty(m)$")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("Einf_vs_m.pdf", dpi=1080)
plt.show()

# combined plot 
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(width_arr, E2_vals,  "b-o",  lw=2, label=r"$E_2(m)$")
ax.loglog(width_arr, Einf_vals, "r--s", lw=2, label=r"$E_\infty(m)$")
plt.xlabel(rf"Width, $m$")
plt.ylabel("Error")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("convergence_width.pdf", dpi=1080)
plt.show()

# ===== F. Effect of Activation Function =====

# params
width_list = [2, 4, 8, 16, 32, 64, 128]
epochs_f = 20000 
restarts_f = 3

# store results
E2_tanh_vals = []
Einf_tanh_vals = []

# optimizer (adam)
optimizer_f = optax.adam(learning_rate=1e-3)

# model initialization for tanh (smaller weights to avoid saturation)
def init_model_tanh(rng, m):
    key_w, key_b, key_a, key_c = jax.random.split(rng, 4)
    
    # smaller input weights for tanh to prevent saturation
    w = 3.0 * jax.random.normal(key_w, (m,))
    
    # biases uniformly across [-1, 1]
    b = jax.random.uniform(key_b, (m,), minval=-1.0, maxval=1.0)
    
    # small output layer weights
    a = 0.1 * jax.random.normal(key_a, (m,))
    
    c = jnp.array(0.0)  # output bias

    return {"w": w, "b": b, "a": a, "c": c}


# training step for tanh 
@jax.jit
def step_f(params, opt_state, t, y):
    loss_val, grads = jax.value_and_grad(mse_loss)(params, t, y, tanh)
    updates, opt_state = optimizer_f.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss_val

# loop through each width
for m in width_list:
    print(f"\n=== width m = {m} ===")

    # store errors from restarts
    err2_list = []
    errinf_list = []

    for r in range(restarts_f):
        print(f" restart {r}")

        # initialize parameters
        key = jax.random.PRNGKey(2000 + 19 * m + r)
        params = init_model_tanh(key, m)
        opt_state = optimizer_f.init(params)

        # training loop
        for epoch in range(epochs_f):
            key_epoch = jax.random.PRNGKey(epoch)
            perm = jax.random.permutation(key_epoch, len(t_train_j))
            t_batch = t_train_j[perm]
            y_batch = y_train_j[perm]
            params, opt_state, train_loss = step_f(params, opt_state, t_batch, y_batch)

        # compute test predictions
        y_pred = model_apply(params, t_test_j, tanh)

        # compute errors (convert to numpy first)
        residual = np.array(y_pred) - np.array(y_test)
        E2 = np.sqrt(np.mean(residual**2))
        Einf = np.max(np.abs(residual))

        err2_list.append(E2)
        errinf_list.append(Einf)

    # take the median across restarts
    E2_tanh_vals.append(np.median(err2_list))
    Einf_tanh_vals.append(np.median(errinf_list))

# convert to numpy arrays
E2_tanh_vals = np.array(E2_tanh_vals)
Einf_tanh_vals = np.array(Einf_tanh_vals)

# log-log slope estimation for tanh
logm = np.log(width_list)
logE2_th = np.log(E2_tanh_vals)
logEin_th = np.log(Einf_tanh_vals)

alpha_E2_tanh, _ = np.polyfit(logm, logE2_th, 1)
alpha_Einf_tanh, _ = np.polyfit(logm, logEin_th, 1)

print("\nestimated slopes (tanh):")
print(f"  alpha_E2_tanh   ~ {-alpha_E2_tanh:.3f}")
print(f"  alpha_Einf_tanh ~ {-alpha_Einf_tanh:.3f}")

# plot E2(m) for tanh
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(width_list, E2_tanh_vals, "b-o", lw=2, label=r"$E_2(m)$")
plt.xlabel(rf"Width, $m$")
plt.ylabel(r"$E_2(m)$")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("E2_tanh_vs_m.pdf", dpi=1080)
plt.show()

# plot Einf(m) for tanh
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(width_list, Einf_tanh_vals, "r--s", lw=2, label=r"$E_\infty(m)$")
plt.xlabel(rf"Width, $m$")
plt.ylabel(r"$E_\infty(m)$")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("Einf_tanh_vs_m.pdf", dpi=1080)
plt.show()

# combined plot 
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(width_arr, E2_tanh_vals,  "b-o",  lw=2, label=r"$E_2(m)$")
ax.loglog(width_arr, Einf_tanh_vals, "r--s", lw=2, label=r"$E_\infty(m)$")
plt.xlabel(rf"Width, $m$")
plt.ylabel("Error")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("convergence_tanh.pdf", dpi=1080)
plt.show()

# combined comparison plot for ReLU and tanh
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(width_arr, E2_vals,      "b-o",  lw=2, label=r"ReLU: $E_2(m)$")
ax.loglog(width_arr, Einf_vals,    "r-s", lw=2, label=r"ReLU: $E_\infty(m)$")
ax.loglog(width_arr, E2_tanh_vals, "b--o", lw=2, label=r"Tanh: $E_2(m)$")
ax.loglog(width_arr, Einf_tanh_vals, "r--s", lw=2, label=r"Tanh: $E_\infty(m)$")
plt.xlabel(rf"Width, $m$")
plt.ylabel("Error")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("convergence_relu_vs_tanh.pdf", dpi=1080)
plt.show()