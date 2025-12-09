import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as random
import optax
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
#from flax import nnx
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

# parameters for plotting
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'cmr10'
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.size'] = 20
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['axes.formatter.use_mathtext'] = True

def generate_data(A, N=1000, seed=None):
    """Generate a noisy time series on {t_i, y_i}_{i=1}^N on [0, 2pi].."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2*np.pi, N)
    dt = (2*np.pi) / (N - 1)
    s_hat = np.sqrt(dt / np.pi) * np.sin(t)    # ||s_hat||_2 = 1
    s = A * s_hat
    n = rng.normal(loc=0.0, scale=1.0, size=N)
    y = s + n
    return t, y, s, n, s_hat, dt

def rho(y, s_hat):
    """detection statistics"""
    return np.dot(y, s_hat)

def sample_rhos(A, M, N=1000):
    """draw M realizations of y at amplitude A and return rhos."""
    rhos = np.zeros(M)
    # s_hat is independent of A, so just grab it once
    _, _, _, _, s_hat, _ = generate_data(A, N=N, seed=0)
    for k in range(M):
        _, y, _, _, _, _ = generate_data(A, N=N, seed=k)
        rhos[k] = rho(y, s_hat)
    return rhos, s_hat

# ===== B. Matched-filter recap and extended study =====
print("\n===== B. Matched-filter recap and extended study =====\n")
print("\n1. Generation of independent realizations of y:\n")
M = 20000
rhos_A0, s_hat = sample_rhos(A=0, M=M)
rhos_A3, _     = sample_rhos(A=3, M=M)

# plot for A=0
fig, ax = plt.subplots(figsize=(15, 6))
ax.hist(rhos_A0, bins=40, density=True, alpha=0.5, color='blue', edgecolor='black', label='Empirical')
xs = np.linspace(-5, 5, 500)
ax.plot(xs, norm.pdf(xs, loc=0, scale=1), 'r-', label=rf'$\mathcal{{N}}\,(0,\,1)$')
plt.xlabel(r"$\rho$")
plt.ylabel("Density")
plt.legend(loc="upper right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("a0_pdf.pdf", dpi=1080)
plt.show()

# plot for A=3
fig, ax = plt.subplots(figsize=(15, 6))
ax.hist(rhos_A3, bins=40, density=True, alpha=0.5, color='blue', edgecolor='black', label='Empirical')
xs = np.linspace(-2, 8, 500)
ax.plot(xs, norm.pdf(xs, loc=3, scale=1), 'r-', label=rf'$\mathcal{{N}}\,(3,\,1)$')
plt.xlabel(r"$\rho$")
plt.ylabel("Density")
plt.legend(loc="upper right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("a3_pdf.pdf", dpi=1080)
plt.show()

print("\n2. Creating filter_classifier and its verification:\n")
def filter_classifier(y, s_hat, rho0):
    rho_val = np.dot(y, s_hat)
    return int(rho_val >= rho0)

# get s_hat (does not depend on A)
t, _, _, _, s_hat, _ = generate_data(A=0, seed=0)

rho0 = 1.5 # threshold

# build dataset to predict class
def build_dataset(M=5000, N=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    labels = []

    for _ in range(M):
        c = rng.integers(0, 2)     # 0 or 1
        A = 0 if c == 0 else 3
        _, y, _, _, _, _ = generate_data(A, N=N, seed=rng.integers(1e9))
        X.append(y)
        labels.append(c)

    return np.array(X, dtype=np.float64), np.array(labels, dtype=np.int32)

X, labels = build_dataset(M=5000, N=1000, seed=0)

rhos = X @ s_hat
preds_theory = (rhos >= rho0).astype(int)

# compute classifier outputs
preds_func = np.array([filter_classifier(y, s_hat, rho0) for y in X])

print("All equal?", np.all(preds_theory == preds_func))
print("Mismatches:", np.sum(preds_theory != preds_func))

# plot for predicted class
fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(rhos, preds_func, s=30, color='blue', edgecolors='black', label='Classifier output')
ax.axvline(rho0, color='red', linestyle='--', linewidth=2, label=rf'$\rho_0 = {rho0}$')
plt.xlabel(r"$\rho$")
plt.ylabel("Predicted class")
plt.yticks([0,1])
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("pred_class.pdf", dpi=1080)
plt.show()

# plot for PDF of predicted class
fig, ax = plt.subplots(figsize=(15, 6))
ax.hist(rhos[preds_func == 0], bins=40, density=True, alpha=0.5, edgecolor='black', color='blue', label='Predicted class 0')
ax.hist(rhos[preds_func == 1], bins=40, density=True, alpha=0.5, edgecolor='black', color='orange', label='Predicted class 1')
ax.axvline(rho0, color='red', linestyle='--', lw=2, label=rf'$\rho_0 = {rho0}$')
plt.xlabel(r"$\rho$")
plt.ylabel("Density")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("pred_class_pdf.pdf", dpi=1080)
plt.show()

# plot comparing theoretical prediction vs. function predictions
fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(np.arange(200), preds_theory[:200], facecolors='none', edgecolors='blue', s=60, linewidths=1.5, label='Theory')
ax.scatter(np.arange(200), preds_func[:200], c='red', marker='x', s=60, linewidths=1.5, label='Function')
plt.xlabel("Sample index")
plt.yticks([0, 1])
plt.ylabel("Class")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("pred_class_comp.pdf", dpi=1080)
plt.show()

# ===== C. Building a labeled dataset for a CNN =====
print("\n===== C. Building a labeled dataset for a CNN =====\n")
def build_dataset(M, N, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    labels = []

    for k in range(M):
        c = rng.integers(0, 2)       # 0 = no signal, 1 = signal
        A = 0 if c == 0 else 3
        _, y, _, _, _, _ = generate_data(A, N=N, seed=rng.integers(1e9))
        X.append(y)
        labels.append(c)

    X = np.array(X, dtype=np.float64) 
    labels = np.array(labels, dtype=np.int32)
    return X, labels

print("\n1. Construction of a label dataset of time series:\n")
N = 1000
M = 20000
X, c = build_dataset(M=M, N=N, seed=0)

print(f"Class balance: {np.sum(c==0)} no-signal, {np.sum(c==1)} signal")
print(f"Percentage signal: {100*np.mean(c):.3f}%")

print("\n2. Split of dataset:\n")
# 80-20 split
X_train, X_test, c_train, c_test = train_test_split(
    X, c,
    test_size=0.20,
    stratify=c,
    random_state=0
)

print("Train:", X_train.shape, c_train.shape)
print("Test :", X_test.shape,  c_test.shape)

print(f"Train class balance: {100*np.mean(c_train):.3f}% signal")
print(f"Test class balance: {100*np.mean(c_test):.3f}% signal")

# convert to JAX array
X_train = jnp.array(X_train, dtype=jnp.float64)
X_test  = jnp.array(X_test,  dtype=jnp.float64)
c_train = jnp.array(c_train)
c_test  = jnp.array(c_test)

# ===== Design and training of a 1D CNN classifier =====
print("\n===== D. Design and training of a 1D CNN classifier =====\n")

# 1D CNN using flax linen
class CNN1D(nn.Module):
    N: int
    
    @nn.compact
    def __call__(self, x):
        # x is (B, N, 1)
        
        # convolutional layer
        x = nn.Conv(features=8, kernel_size=(16,), padding="SAME")(x)
        
        # relu
        x = nn.relu(x)
        
        # max pooling
        x = nn.max_pool(x, window_shape=(4,), strides=(4,))
        
        # flattern
        x = x.reshape((x.shape[0], -1))  # (B, 2000)
        
        # one dense layer
        x = nn.Dense(32)(x)
        x = nn.relu(x)
        
        # final output layer
        logits = nn.Dense(2)(x)
        
        return logits

print("\n1. 1D CNN architecture parameter counts:\n")
# functions for trainable parameter count
def count_params(params):
    return sum(p.size for p in jax.tree_util.tree_leaves(params))

def detailed_param_count(params):
    result = {}

    def traverse(tree, parent=""):
        for key, val in tree.items():
            name = f"{parent}/{key}" if parent else key
            if isinstance(val, dict):
                traverse(val, name)
            else:
                result.setdefault(parent, 0)
                result[parent] += val.size

    traverse(unfreeze(params))
    return result

# initialize model and parameters
rng = jax.random.PRNGKey(0)
dummy = jnp.zeros((1, 1000, 1))   # NOTE: must be (B, N, 1)

model = CNN1D(N=1000)
params = model.init(rng, dummy)

# parameter counts
total = count_params(params)
print("Total parameters:", total)

details = detailed_param_count(params)
for layer, n in details.items():
    print(f"{layer:20s} : {n}")
    
print("\n2. Normalize and reshape the data:\n")
# normalize the data
X_train_np = np.array(X_train)
X_test_np  = np.array(X_test)

mean = X_train_np.mean()
std  = X_train_np.std()

X_train_np = (X_train_np - mean) / std
X_test_np  = (X_test_np  - mean) / std

# reshape for Conv1D
X_train_np = X_train_np[:, :, None]
X_test_np  = X_test_np[:, :, None]

print("\n3. Split for validation and training:\n")
# 80-20 train-validation split
X_tr_np, X_val_np, c_tr_np, c_val_np = train_test_split(
        X_train_np, np.array(c_train), test_size=0.2, 
        stratify=np.array(c_train), random_state=1)

# convert to JAX arrays
X_tr  = jnp.array(X_tr_np)
X_val = jnp.array(X_val_np)
X_te  = jnp.array(X_test_np)

c_tr  = jnp.array(c_tr_np)
c_val = jnp.array(c_val_np)
c_te  = jnp.array(c_test)

print("\nTraining the 1D CNN:\n")

def loss_fn(params, batch_x, batch_y):
    logits = model.apply(params, batch_x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()
    return loss, logits

def accuracy(logits, labels):
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == labels)

optimizer = optax.adamw(learning_rate=1e-3, weight_decay=5e-4)

@jax.jit
def train_step(params, opt_state, xb, yb):
    (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, xb, yb
    )
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    acc = accuracy(logits, yb)
    return params, opt_state, loss, acc


@jax.jit
def eval_step(params, xb, yb):
    loss, logits = loss_fn(params, xb, yb)
    acc = accuracy(logits, yb)
    return loss, acc

# initialize the model and optimizer
model = CNN1D(N=N)
rng = jax.random.PRNGKey(23)
params = model.init(rng, X_tr[:8])   # dummy batch for shapes
opt_state = optimizer.init(params)

# training params
batch_size = 256   # "None" for full batch training
epochs = 20

train_losses = []
train_accs = []
val_losses = []
val_accs = []

patience = 3         
best_val_loss = jnp.inf    
patience_counter = 0       
params_best = None         

# training loop
for ep in range(epochs):
    
    # reproducible shuffle
    rng, sub = jax.random.split(rng)
    perm = jax.random.permutation(sub, X_tr.shape[0])
    X_shuf = X_tr[perm]
    c_shuf = c_tr[perm]

    # batching mode
    if batch_size is None:
        """full-batch"""
        xb = X_shuf
        yb = c_shuf

        params, opt_state, loss, acc = train_step(params, opt_state, xb, yb)

        ep_loss = float(loss)
        ep_acc  = float(acc)

    else:
        """mini-batch"""
        num_batches = X_shuf.shape[0] // batch_size

        ep_loss = 0.0
        ep_acc  = 0.0

        for i in range(num_batches):
            xb = X_shuf[i*batch_size:(i+1)*batch_size]
            yb = c_shuf[i*batch_size:(i+1)*batch_size]

            params, opt_state, loss, acc = train_step(
                params, opt_state, xb, yb
            )
            ep_loss += float(loss)
            ep_acc  += float(acc)

        ep_loss /= num_batches
        ep_acc  /= num_batches

    # validation
    val_loss, val_acc = eval_step(params, X_val, c_val)

    train_losses.append(ep_loss)
    train_accs.append(ep_acc)
    val_losses.append(float(val_loss))
    val_accs.append(float(val_acc))

    print(f"Epoch {ep+1:2d} | "
          f"train_loss={ep_loss:.4f}, train_acc={ep_acc:.4f} | "
          f"val_loss={float(val_loss):.4f}, val_acc={float(val_acc):.4f}")
    
    # early stopping check
    if float(val_loss) < float(best_val_loss) - 1e-6:  
        best_val_loss = float(val_loss)
        params_best = params
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"\nEarly stopping triggered at epoch {ep+1}")
        params = params_best
        break

# final test loss and accuracy
test_loss, test_acc = eval_step(params, X_te, c_te)
print("\nTest Loss:", float(test_loss))
print("Test Accuracy:", float(test_acc))

print("\n4. Plot training and validation loss:\n")
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(train_losses, "b-", lw=2, label="Training")
ax.plot(val_losses, "r--", lw=2, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax.legend(frameon=False)
ax.tick_params(axis="both", which="both", direction="in")
plt.savefig("cnn_loss.pdf", dpi=1080)
plt.show()

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(train_accs, "b-", lw=2, label="Training")
ax.plot(val_accs, "r--", lw=2, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("cnn_acc.pdf", dpi=1080)
plt.show()

print("\n5. CNN's performance on the held out test set:\n")
test_logits = model.apply(params, X_te)
test_probs  = jax.nn.softmax(test_logits, axis=-1)[:, 1]
test_preds  = (test_probs >= 0.5).astype(int)

test_acc = jnp.mean(test_preds == c_te)

print("\nTest accuracy:", float(test_acc))

# ===== E. Quantitative comparison: CNN vs matched filter =====
print("\n===== E. Quantitative comparison: CNN vs matched filter =====\n")

print("\n1. Construct ROC:\n")

# matched-filter ROC 
X_te_flat = np.array(X_te[..., 0]) 

rho_vals = X_te_flat @ s_hat   

rho_range = np.linspace(rho_vals.min(), rho_vals.max(), 400)
tpr_mf = []
fpr_mf = []

for rho0 in rho_range:
    preds = (rho_vals >= rho0).astype(int)

    tp = np.sum((preds == 1) & (c_te == 1))
    fp = np.sum((preds == 1) & (c_te == 0))
    fn = np.sum((preds == 0) & (c_te == 1))
    tn = np.sum((preds == 0) & (c_te == 0))

    tpr_mf.append(tp / (tp + fn))
    fpr_mf.append(fp / (fp + tn))

auc_mf = auc(fpr_mf, tpr_mf)
"""
fpr_mf, tpr_mf, mf_thresholds = roc_curve(c_te, rho_vals)
auc_mf = auc(fpr_mf, tpr_mf)
"""
# CNN ROC
"""
def predict_cnn_proba(params, X):
    logits = model.apply(params, X)
    probs = jax.nn.softmax(logits, axis=-1)
    return np.array(probs[:, 1])   # probability of class "1" (signal)

cnn_probs = predict_cnn_proba(params, X_te)

fpr_cnn, tpr_cnn, cnn_thresholds = roc_curve(c_te, cnn_probs)
auc_cnn = auc(fpr_cnn, tpr_cnn)
"""
logits_te = model.apply(params, X_te)         
probs = jax.nn.softmax(logits_te, axis=-1)    
p_hat = np.array(probs[:, 1])                 

p_range = np.linspace(0, 1, 400)
tpr_cnn, fpr_cnn = [], []

for p0 in p_range:
    preds = (p_hat >= p0).astype(int)

    tp = np.sum((preds == 1) & (c_te == 1))
    fp = np.sum((preds == 1) & (c_te == 0))
    tn = np.sum((preds == 0) & (c_te == 0))
    fn = np.sum((preds == 0) & (c_te == 1))

    tpr_cnn.append(tp / (tp + fn))
    fpr_cnn.append(fp / (fp + tn))
    
print("\n2. Plot ROC:\n")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(fpr_mf, tpr_mf, 'b-', lw=2, label=f"Matched Filter (AUC = {auc_mf:.4f})")
ax.plot(fpr_cnn, tpr_cnn, 'r--', lw=2, label=f"CNN (AUC = {auc_cnn:.4f})")
ax.plot([0, 1], [0, 1], 'k--', lw=1) 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("roc.pdf", dpi=1080)
plt.show()

# ===== F. Going further =====
print("\n===== F. Going further =====\n")

print("\nInterpreting the learned convolutional filters:\n")

p = params['params']

print("\nAvailable layers:", list(p.keys()))

# first convolutional layer
k1 = np.array(p['Conv_0']['kernel'])
print(f"Conv_0 kernel shape: {k1.shape}\n")  
K1 = k1.shape[0] # kernel size
n_filters_1 = k1.shape[2] # number of learned filters

# extract each filter (input channel = 1)
kernels_conv1 = [k1[:, 0, j] for j in range(n_filters_1)]

# construct the known matched-filter template at same resolution
t_kernel = np.linspace(0, 2*np.pi, K1)
dt_kernel = 2*np.pi / (K1 - 1)
s_hat_kernel = np.sqrt(dt_kernel / np.pi) * np.sin(t_kernel)

# normalize for plotting
s_hat_norm = s_hat_kernel / np.max(np.abs(s_hat_kernel))

# plot for learned kernel and $\hat{s}$
fig, ax = plt.subplots(figsize=(15, 6))
for j, k_j in enumerate(kernels_conv1):
    k_norm = k_j / (np.max(np.abs(k_j)) + 1e-12)
    ax.plot(k_norm, lw=2, label=f"Filter {j}")
ax.plot(s_hat_norm, 'k--', lw=3, label="Template $\hat{s}$")
ax.axhline(0, color='gray', linestyle=':', linewidth=0.7)
plt.xlabel("Kernel index")
plt.ylabel("Normalized amplitude")
plt.legend(ncol=3, frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("comp_kernel.pdf", dpi=1080)
plt.show()

# correlation between learned filters and $\hat{s}$
s_norm = s_hat_kernel / np.linalg.norm(s_hat_kernel)

correlations = []
for j, k_j in enumerate(kernels_conv1):
    k_norm = k_j / (np.linalg.norm(k_j) + 1e-12)
    corr = np.dot(k_norm, s_norm)
    correlations.append(corr)
    print(f"Filter {j}: correlation = {corr:+.4f}")

best_idx = np.argmax(np.abs(correlations))
best_corr = correlations[best_idx]

print(f"\nBest matching filter: Filter {best_idx}")
print(f"Absolute correlation: {abs(best_corr):.4f}")

print("\nDifferent signal strengths:\n")

# function for multi-amplitude dataset
def build_multiamp_dataset(M=20000, N=1000, seed=0):
    rng = np.random.default_rng(seed)
    X = []
    labels = []
    amplitudes = []
    
    for k in range(M):
        has_signal = rng.integers(0, 2)  # 50-50 split
        
        if has_signal == 0:
            A = 0
            label = 0  # no signal
        else:
            A = rng.choice([1, 3, 10])  # random signal strength
            label = 1  # signal present (A >= 1)
        
        _, y, _, _, _, _ = generate_data(A, N=N, seed=rng.integers(int(1e9)))
        X.append(y)
        labels.append(label)
        amplitudes.append(A)
    
    return np.array(X), np.array(labels), np.array(amplitudes)

X_multi, c_multi, A_multi = build_multiamp_dataset(M=20000, N=1000, seed=42)

# check distribution
print(f"\nDataset distribution:")
print(f"A=0: {np.sum(A_multi == 0)} ({100*np.mean(A_multi == 0):.1f}%)")
print(f"A=1: {np.sum(A_multi == 1)} ({100*np.mean(A_multi == 1):.1f}%)")
print(f"A=3: {np.sum(A_multi == 3)} ({100*np.mean(A_multi == 3):.1f}%)")
print(f"A=10: {np.sum(A_multi == 10)} ({100*np.mean(A_multi == 10):.1f}%)")

# 80-20 split
X_train_ma, X_test_ma, c_train_ma, c_test_ma, A_train_ma, A_test_ma = train_test_split(
    X_multi, c_multi, A_multi,
    test_size=0.2,
    stratify=c_multi,
    random_state=0
)

# normalize test data
X_test_ma_norm = (X_test_ma - mean) / std
X_test_ma_jax = jnp.array(X_test_ma_norm, dtype=jnp.float64)[:, :, None]

# CNN predictions
cnn_logits_ma = model.apply(params, X_test_ma_jax)
cnn_probs_ma = np.array(jax.nn.softmax(cnn_logits_ma, axis=-1)[:, 1])

# matched filter predictions
rhos_test_ma = X_test_ma @ s_hat

# performance of CNN
fpr_cnn_all, tpr_cnn_all, _ = roc_curve(c_test_ma, cnn_probs_ma)
auc_cnn_all = auc(fpr_cnn_all, tpr_cnn_all)

# performance of matched filter
fpr_mf_all, tpr_mf_all, _ = roc_curve(c_test_ma, rhos_test_ma)
auc_mf_all = auc(fpr_mf_all, tpr_mf_all)

print(f"\nCNN AUC: {auc_cnn_all:.4f}")
print(f"Matched Filter AUC: {auc_mf_all:.4f}")

fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(fpr_mf_all, tpr_mf_all, 'b-', lw=2, label=f'Matched Filter (AUC={auc_mf_all:.4f})')
ax.plot(fpr_cnn_all, tpr_cnn_all, 'r--', lw=2, label=f'CNN (AUC={auc_cnn_all:.4f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig("comp_allamp.pdf", dpi=1080)
plt.show()

# per-amplitude analysis
for A_val in [1, 3, 10]:
    # include both signal at this amplitude and all no-signal examples
    mask = (A_test_ma == A_val) | (A_test_ma == 0)
    
    print(f"\nFor A={A_val}:")
    print(f"Total samples: {mask.sum()}")
    print(f"A=0 (no signal): {np.sum(A_test_ma[mask] == 0)}")
    print(f"A={A_val} (signal): {np.sum(A_test_ma[mask] == A_val)}")
    
    # CNN ROC and AUC
    fpr_cnn, tpr_cnn, _ = roc_curve(c_test_ma[mask], cnn_probs_ma[mask])
    auc_cnn = auc(fpr_cnn, tpr_cnn)
    
    # matched filter ROC and AUC
    fpr_mf, tpr_mf, _ = roc_curve(c_test_ma[mask], rhos_test_ma[mask])
    auc_mf = auc(fpr_mf, tpr_mf)
    
    print(f"CNN AUC: {auc_cnn:.4f}")
    print(f"Matched Filter AUC: {auc_mf:.4f}")
    print(f"Difference: {abs(auc_cnn - auc_mf):.4f}")
    
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(fpr_mf, tpr_mf, 'b-', lw=2, label=f'Matched Filter (AUC={auc_mf:.4f})')
    ax.plot(fpr_cnn, tpr_cnn, 'r--', lw=2, label=f'CNN (AUC={auc_cnn:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(frameon=False)
    plt.tick_params(axis="both", which="both", direction="in")
    plt.savefig(f"comp_amp{A_val}.pdf", dpi=1080)
    plt.show()