import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import jax
jax.config.update("jax_enable_x64", True)
import optax
import jax.random as random
from flax import linen as nn
import jax.numpy as jnp
from typing import Sequence

# ===== B. Running the starter code & dataset generation =====
print("\n===== B. Running the starter code & dataset generation =====\n")

def sho_analytic(t, A=1.0, k=1.0):
    """ Analytic solution for the 1D harmonic oscillator.

    Initial conditions are
    q(0) = A
    p(0) = 0
    
    Parameters are
    mass m = 1
    spring constant k > 0
    """

    omega = np.sqrt(k)
    q     = A * np.cos(omega*t)
    p     = -A*omega*np.sin(omega*t)
    return q, p

def sho_rhs(q, p, k=1.0):
    """Right-hand side of the 
    
    dq/dt = p
    dp/dt = -k q
    """
    
    dqdt = p
    dpdt = -k * q
    return dqdt, dpdt

def rk4_step(q, p, dt, k=1.0):
    """Single RK4 step taking us from time t to t+dt"""
    k1_q, k1_p = sho_rhs(q, p, k)
    k2_q, k2_p = sho_rhs(q + 0.5*dt*k1_q, p + 0.5*dt*k1_p, k)
    k3_q, k3_p = sho_rhs(q + 0.5*dt*k2_q, p + 0.5*dt*k2_p, k)
    k4_q, k4_p = sho_rhs(q + dt*k3_q, p + dt*k3_p, k)

    q_next = q + (dt/6.0) * (k1_q + 2*k2_q + 2*k3_q + k4_q)
    p_next = p + (dt/6.0) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
    return q_next, p_next

def sho_numerical(A=1.0, k=1.0, T=10.0, dt=0.01):
    """ Numerical solution of the harmonic oscillator. """
    
    t = np.arange(0.0, T + dt, dt)
    q = np.zeros_like(t)
    p = np.zeros_like(t)

    # initial conditions
    q[0] = A
    p[0] = 0.0

    for n in range(len(t) - 1):
        q[n+1], p[n+1] = rk4_step(q[n], p[n], dt, k)

    return t, q, p

print("\n1. Compute exact and numerical p, q:\n")

times, q_num, p_num = sho_numerical(A=1.0, k=1.0, T=10.0, dt=0.01)
q_exact, p_exact    = sho_analytic(times, A=1.0, k=1.0)

# parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 22
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(times,q_num, 'b-', lw=3, label=r'$q_{num}(t)$')
ax.plot(times,q_exact, 'r--', lw=3, label=r'$q_{exact}(t)$')
ax.plot(times,p_num, 'g-', lw=3, label=r'$p_{num}(t)$')
ax.plot(times,p_exact, '--', color="orange", lw=3, label=r'$p_{exact}(t)$')
plt.xlabel(r"$t$")
plt.ylabel(r"$p(t), q(t)$")
plt.xlim(0,10)
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("pq_exact_num.pdf", dpi=1080)
plt.show()

# plot error
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(times, (q_num - q_exact)*1e10, "k-", lw=3)
plt.xlabel(r'$t$')
plt.ylabel(r'$e(t)\times10^{10}$')
plt.xlim(0,10)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("q_exact_num_err.pdf", dpi=1080)
plt.show()

max_err = np.max(np.abs(q_num - q_exact))
print(f"Maximum absolute error = {max_err:.3e}")

print("\n2. Data generation:\n")

k = 1.0
A_list = [0.5, 1.0, 1.5, 2.0]
T = 10.0
dt = 0.01

times = np.arange(0.0, T + dt, dt)  # shape (1001,)
all_q = []
all_p = []

for A in A_list:
    q_A, p_A = sho_analytic(times, A=A, k=k)
    all_q.append(q_A)
    all_p.append(p_A)

all_q = np.stack(all_q, axis=0)  # shape (m, N_t)
all_p = np.stack(all_p, axis=0)  # shape (m, N_t)

# flatten trajectories over all amplitudes and times
q_flat = all_q.reshape(-1)  # shape (4004,)
p_flat = all_p.reshape(-1)  # shape (4004,)

# use analytic expression (qdot = p_exact,  pdot = -k * q_exact)
qdot_flat = p_flat.copy()
pdot_flat = -k * q_flat

# inputs X = (q, p), outputs y = (qdot, pdot)
X = np.stack([q_flat,   p_flat],   axis=1)  # shape (4004, 2)
Y = np.stack([qdot_flat, pdot_flat], axis=1)  # shape (4004, 2)

print("X shape:", X.shape)
print("Y shape:", Y.shape)

rng = np.random.default_rng(seed=0)

N = X.shape[0] 
perm = rng.permutation(N)

# 80-20 split
train_frac = 0.8
N_train = int(train_frac * N) 
train_idx = perm[:N_train]
val_idx   = perm[N_train:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val,   Y_val   = X[val_idx],   Y[val_idx]

print("Train size:", X_train.shape[0])
print("Val size  :", X_val.shape[0])

X_train_jax = jnp.asarray(X_train)
Y_train_jax = jnp.asarray(Y_train)
X_val_jax   = jnp.asarray(X_val)
Y_val_jax   = jnp.asarray(Y_val)

# ===== C. Learning dynamics with a neural Hamiltonian =====
print("\n===== C. Learning dynamics with a neural Hamiltonian =====\n")

# NN architecture
class HamiltonianNN(nn.Module):
    hidden: Sequence[int] = (32,)

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float64)
        for h in self.hidden:
            x = nn.tanh(nn.Dense(h, dtype=jnp.float64)(x))
            #x = nn.relu(nn.Dense(h, dtype=jnp.float64)(x))
        return nn.Dense(1, dtype=jnp.float64)(x)

# Hamiltonian vector fields' prediction
def hamiltonian_vector_field(params, model, qp):

    def H_single(z):
        return model.apply(params, z[None,:]).sum()

    grads = jax.vmap(jax.grad(H_single))(qp)
    dH_dq = grads[:,0:1]
    dH_dp = grads[:,1:2]

    return jnp.concatenate([dH_dp, -dH_dq], axis=1)

# loss function
def loss_fn(params, model, X, Y):
    Y_pred = hamiltonian_vector_field(params, model, X)
    return jnp.mean((Y_pred - Y)**2)

# lbfgs as optimizer
optimizer = optax.lbfgs()

model = HamiltonianNN()
key = random.PRNGKey(0)
params = model.init(key, X_train_jax[:1])
opt_state = optimizer.init(params)

def lbfgs_step(params, opt_state, X, Y):

    # compute loss and gradient
    loss, grad = jax.value_and_grad(loss_fn)(params, model, X, Y)

    # update call
    updates, opt_state = optimizer.update(
        grad,
        opt_state,
        params=params,
        value=loss,
        grad=grad,
        value_fn=lambda p: loss_fn(p, model, X, Y)
    )

    # apply update
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# params
num_epochs = 500
train_losses = []
val_losses = []

# training loop
for epoch in range(1, num_epochs+1):

    params, opt_state, train_loss = lbfgs_step(
        params, opt_state, X_train_jax, Y_train_jax
    )

    val_loss = loss_fn(params, model, X_val_jax, Y_val_jax)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch {epoch}: train_loss={train_loss:.6e}, val_loss={val_loss:.6e}")
        
# plot training and validation losses
fig, ax = plt.subplots(figsize=(15, 6))
ax.semilogy(train_losses, "b-", lw=3, label="Training")
ax.semilogy(val_losses, "r--", lw=3, label="Validation")
plt.xlabel(r"Epoch")
plt.ylabel(r"$MSE$")
plt.xlim(0, 500)
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("hamilton_loss.pdf", dpi=1080)
plt.show()

# ===== D. Using your new model =====
print("\n===== D. Using your new model =====\n")

A_test = 1.25

def learned_rhs(params, model, q, p):
    qp = jnp.array([[q, p]])
    qdot, pdot = hamiltonian_vector_field(params, model, qp)[0]
    return float(qdot), float(pdot)

def rk4_step_learned(q, p, dt, params, model):
    k1_q, k1_p = learned_rhs(params, model, q, p)
    k2_q, k2_p = learned_rhs(params, model, q + 0.5*dt*k1_q, p + 0.5*dt*k1_p)
    k3_q, k3_p = learned_rhs(params, model, q + 0.5*dt*k2_q, p + 0.5*dt*k2_p)
    k4_q, k4_p = learned_rhs(params, model, q + dt*k3_q, p + dt*k3_p)
    q_next = q + (dt/6.0)*(k1_q + 2*k2_q + 2*k3_q + k4_q)
    p_next = p + (dt/6.0)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
    return q_next, p_next

def integrate_learned(A_test, T=10.0, dt=0.01):
    t = np.arange(0.0, T+dt, dt)
    q = np.zeros_like(t); p = np.zeros_like(t)
    q[0] = A_test; p[0] = 0.0
    for n in range(len(t)-1):
        q[n+1], p[n+1] = rk4_step_learned(q[n], p[n], dt, params, model)
    return t, q, p

# integrate and compare to analytic solution
t, q_learn, p_learn = integrate_learned(A_test, T=10.0, dt=0.01)
q_exact, p_exact    = sho_analytic(t, A=A_test, k=1.0)

# plot trajectories
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t, q_exact, 'b-', lw=3, label=r'$q_{exact}(t)$')
ax.plot(t, q_learn, 'r--', lw=3, label=r'$q_\theta(t)$')
ax.plot(t, p_exact, 'g-', lw=3, label=r'$p_{exact}(t)$')
ax.plot(t, p_learn, '--', color="orange", lw=3, label=r'$p_\theta(t)$')
plt.xlabel(r'$t$')
plt.ylabel(r'$p, q$')
plt.xlim(0,10)
plt.ylim(-1.5,1.5)
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("learned_pq.pdf", dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t, q_learn - q_exact, 'b-', lw=3, label=r'$q_\theta(t) - q_{exact}(t)$')
ax.plot(t, p_learn - p_exact, 'r--', lw=3, label=r'$p_\theta(t) - p_{exact}(t)$')
plt.xlabel(r'$t$')
plt.ylabel(r'$e(t)$')
plt.xlim(0,10)
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("learned_pq_err.pdf", dpi=1080)
plt.show()

# extrapolate
T_long = 100.0
t_long, q_learn_long, p_learn_long = integrate_learned(A_test, T=T_long, dt=0.01)
q_exact_long, p_exact_long = sho_analytic(t_long, A=A_test, k=1.0)

# plot trajectories over long period
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t_long, q_exact_long, 'b-', lw=3, label=r'$q_{exact}$')
ax.plot(t_long, q_learn_long, 'r--', lw=3, label=r'$q_\theta$')
ax.plot(t_long, p_exact_long, 'g-', lw=3, label=r'$p_{exact}(t)$')
ax.plot(t_long, p_learn_long, '--', color="orange", lw=3, label=r'$p_\theta(t)$')
plt.xlabel(r'$t$')
plt.ylabel(r'$p, q$') 
plt.xlim(t_long.min(), t_long.max())
plt.ylim(-1.75,1.75)
plt.legend(loc="upper left", ncol=4, frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("learned_pq_ex.pdf", dpi=1080)
plt.show()

# plot error vs time
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t_long, q_learn_long - q_exact_long, "b-", lw=3, label=r'$q_\theta(t) - q_{exact}(t)$')
ax.plot(t_long, p_learn_long - p_exact_long, "r--", lw=3, label=r'$p_\theta(t) - p_{exact}(t)$')
plt.xlabel(r'$t$'); 
plt.ylabel(r'$e(t)$')
plt.xlim(t_long.min(), t_long.max())
plt.ylim(-0.01,0.01)
plt.legend(loc="lower left", ncol=2, frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("learned_pq_ex_err.pdf", dpi=1080)
plt.show()

# ===== E. Going further =====
print("\n===== E. Going further =====\n")
# conservation-based physics loss
def loss_fn_phys(params, model, X_batch, Y_batch, lam=1e-5):
    Y_pred = hamiltonian_vector_field(params, model, X_batch)
    data_loss = jnp.mean((Y_pred - Y_batch)**2)

    # energy conservation loss: H(q_{t+1},p_{t+1}) - H(q_t,p_t) should be 0
    X_t   = X_batch[:-1]    
    X_tp1 = X_batch[1:]     

    H_t   = model.apply(params, X_t)
    H_tp1 = model.apply(params, X_tp1)

    energy_conservation = jnp.mean((H_tp1 - H_t)**2)

    return data_loss + lam * energy_conservation

def train_step_phys(params, opt_state, Xb, Yb):
    loss, grads = jax.value_and_grad(loss_fn_phys)(params, model, Xb, Yb)

    updates, opt_state = optimizer.update(
        grads,
        opt_state,
        params=params,
        value=loss,
        grad=grads,
        value_fn=lambda p: loss_fn_phys(p, model, Xb, Yb)
    )
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


# retrain model
num_epochs = 500
train_losses = []
val_losses = []

params = model.init(random.PRNGKey(99), X_train_jax[:1])
opt_state = optimizer.init(params)

for epoch in range(1, num_epochs+1):
    params, opt_state, train_loss = train_step_phys(
        params, opt_state, X_train_jax, Y_train_jax
    )
    val_loss = loss_fn_phys(params, model, X_val_jax, Y_val_jax)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if epoch == 1 or epoch % 50 == 0:
        print(f"Epoch {epoch}: train={train_loss:.3e}, val={val_loss:.3e}")
        
# plot physics-informed training performance
fig, ax = plt.subplots(figsize=(15, 6))
ax.semilogy(train_losses, "b-", lw=3, label="Training")
ax.semilogy(val_losses, "r--", lw=3, label="Validation")
plt.xlabel(r"Epoch")
plt.ylabel(r"$MSE$")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("hamilton_loss_phys.pdf", dpi=1080)
plt.show()

# evaluate for new amplitude
A_test = 1.25 

def learned_rhs(params, model, q, p):
    qp = jnp.array([[q, p]])
    qdot, pdot = hamiltonian_vector_field(params, model, qp)[0]
    return float(qdot), float(pdot)

def rk4_step_learned(q, p, dt):
    k1_q, k1_p = learned_rhs(params, model, q, p)
    k2_q, k2_p = learned_rhs(params, model, q + 0.5*dt*k1_q, p + 0.5*dt*k1_p)
    k3_q, k3_p = learned_rhs(params, model, q + 0.5*dt*k2_q, p + 0.5*dt*k2_p)
    k4_q, k4_p = learned_rhs(params, model, q + dt*k3_q, p + dt*k3_p)
    return (
        q + (dt/6)*(k1_q + 2*k2_q + 2*k3_q + k4_q),
        p + (dt/6)*(k1_p + 2*k2_p + 2*k3_p + k4_p)
    )

def integrate_learned(A, T, dt=0.01):
    t = np.arange(0,T+dt,dt)
    q = np.zeros_like(t); p = np.zeros_like(t)
    q[0] = A; p[0] = 0
    for n in range(len(t)-1):
        q[n+1], p[n+1] = rk4_step_learned(q[n], p[n], dt)
    return t, q, p

# for T = 10
t10, q_learn10, p_learn10 = integrate_learned(A_test, T=10)
q_exact10, p_exact10 = sho_analytic(t10, A=A_test)

# plot predicted trajectories
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t10, q_exact10, 'b-', lw=3, label=r'$q_{exact}(t)$')
ax.plot(t10, q_learn10, 'r--', lw=3, label=r'$q_\theta(t)$')
ax.plot(t10, p_exact10, 'g-', lw=3, label=r'$p_{exact}(t)$')
ax.plot(t10, p_learn10, '--', color="orange", lw=3, label=r'$p_\theta(t)$')
plt.xlabel(r'$t$')
plt.ylabel(r'$p, q$')
plt.xlim(t10.min(), t10.max())
plt.ylim(-1.5,1.5)
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("learned_pq_phys.pdf", dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t, q_learn - q_exact, 'b-', lw=3, label=r'$q_\theta(t) - q_{exact}(t)$ (w/o physics)')
ax.plot(t, q_learn10 - q_exact10, 'r--', lw=3, label=r'$q_\theta(t) - q_{exact}(t)$ (w/ physics)')
ax.plot(t, p_learn - p_exact, 'g-', lw=3, label=r'$p_\theta(t) - p_{exact}(t)$ (w/o physics)')
ax.plot(t, p_learn10 - p_exact10, '--', color="orange", lw=3, label=r'$p_\theta(t) - p_{exact}(t)$ (w/ physics)')
plt.xlabel(r'$t$')
plt.ylabel(r'$e(t)$')
plt.xlim(t10.min(), t10.max())
plt.legend(ncol=2, frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("learned_pq_err_phys.pdf", dpi=1080)
plt.show()

# extrapolation
t_long100, q_learn_long100, p_learn_long100 = integrate_learned(A_test, T=100)
q_exact_long100, p_exact_long100 = sho_analytic(t_long100, A=A_test)

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t_long100, q_exact_long100, 'b-', lw=3, label=r'$q_{exact}(t)$')
ax.plot(t_long100, q_learn_long100, 'r--', lw=3, label=r'$q_\theta(t)$')
ax.plot(t_long100, p_exact_long100, 'g-', lw=3, label=r'$p_{exact}(t)$')
ax.plot(t_long100, p_learn_long100, '--', lw=3, color="orange", label=r'$p_\theta(t)$')
plt.xlabel(r'$t$'); 
plt.ylabel(r'$p, q$')
plt.xlim(t_long100.min(),t_long100.max())
plt.ylim(-1.75,1.75)
plt.legend(ncol=4, frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("learned_pq_ex_phys.pdf", dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(t_long, q_learn_long - q_exact_long, 'b-', lw=3, label=r'$q_\theta(t) - q_{exact}(t)$ (w/o physics)')
ax.plot(t_long100, q_learn_long100 - q_exact_long100, 'r--', lw=3, label=r'$q_\theta(t) - q_{exact}(t)$ (w/ physics)')
ax.plot(t_long, p_learn_long - p_exact_long, 'g-', lw=3, label=r'$p_\theta(t) - p_{exact}(t)$ (w/o physics)')
ax.plot(t_long100, p_learn_long100 - p_exact_long100, '--', color="orange", lw=3, label=r'$p_\theta(t) - p_{exact}(t)$ (w/ physics)')
plt.xlabel(r'$t$'); 
plt.ylabel(r'$e(t)$')
plt.xlim(t_long100.min(),t_long100.max())
plt.legend(ncol=2, frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("learned_pq_ex_err_phys.pdf.pdf", dpi=1080)
plt.show()