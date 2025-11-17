import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
import jax.numpy as jnp
import optax

# target function f*
def f_star(t):
    return np.exp(-3.0 * t) * np.sin(8 * np.pi * t)

# train and test sets
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

# relu activation
def relu(x):
    return jnp.maximum(0.0, x)

# model initialization
def init_model(rng, m):
    # Initialize weights and biases for a single layer
    key_w, key_t, key_a, key_c = jax.random.split(rng, 4)
    w = 30.0 * jax.random.normal(key_w, (m,))
    t0 = jax.random.uniform(key_t, (m,), minval=0.0, maxval=1.0)  # kink locations
    b = -w * t0
    a = 0.1 * jax.random.normal(key_a, (m,))
    c = jnp.array(0.0)  # output bias
    return {"w": w, "b": b, "a": a, "c": c}

# forward pass
def model_apply(params, t, activation):
    z = params["w"] * t[:, None] + params["b"]  
    h = activation(z)  # apply activation
    y_pred = jnp.dot(h, params["a"]) + params["c"]  
    return y_pred

# mse loss function
def mse_loss(params, t, y, activation):
    y_pred = model_apply(params, t, activation)
    r = y_pred - y
    return jnp.mean(r * r)

# width and epochs
m = 64  
epochs_d = 500000  

# optimizer (adam)
optimizer_d = optax.adam(learning_rate=1e-4)

# initialize parameters
rng = jax.random.PRNGKey(299)
params_d = init_model(rng, m)

opt_state_d = optimizer_d.init(params_d)

# activation outside the jit
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

# full batch training loop
for epoch in range(epochs_d):
    params_d, opt_state_d, train_loss = step(params_d, opt_state_d, t_train_j, y_train_j)

    train_hist_d.append(float(train_loss))
    
    # compute test mse
    test_loss = mse_loss(params_d, t_test_j, y_test_j, activation)
    test_hist_d.append(float(test_loss))

    if epoch % 5000 == 0:
        print(f"epoch {epoch}, train mse = {train_loss:.4e}, test mse = {test_loss:.4e}")

print(f"\nfinal train mse = {train_hist_d[-1]:.4e}")
print(f"final test  mse = {test_hist_d[-1]:.4e}")

# plot mse vs. epoch
fig, ax = plt.subplots(figsize=(15, 6))
ax.loglog(train_hist_d, "b-", lw=2, label="Train MSE")
ax.loglog(test_hist_d,  "r--", lw=2, label="Test MSE")
ax.axhline(y=1e-4, color='g', linestyle='--', label=r"$10^{-4}$ MSE Threshold")

plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"param_count_mse.pdf", dpi=1080)
plt.show()

# plot model and true function
y_pred_d = model_apply(params_d, t_test_j, activation)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t_test, y_test,     "k-", lw=2, label=r"$f^*(t)$")
ax.plot(t_test, y_pred_d,   "r--", lw=2, label=r"$\tilde f(t)$, m=64")
plt.xlabel(r"$t$")
plt.ylabel(r"$f(t)$")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"model_true_bonus.pdf", dpi=1080)
plt.show()

# count the number of trainable parameters
def count_params_pytree(params):
    leaves = jax.tree_util.tree_leaves(params)  # Collect all leaf arrays from the nested params tree
    sizes = [int(jnp.size(x)) for x in leaves]  # Get the size of each array
    total = sum(sizes)  # Get the total number of trainable scalars
    return total

num_params = count_params_pytree(params_d)
print(f"Total number of trainable parameters: {num_params}")