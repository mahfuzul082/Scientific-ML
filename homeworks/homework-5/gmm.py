from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse

# ===== B. Setup and notation =====
def sample_2d_gaussian(mu, Sigma, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n)


# ===== C. Data for this assignment =====
# setting parameters
rng = np.random.default_rng(29)
n = 3000

# ground-truth
w_true = np.array([0.2, 0.8])
mu1 = np.array([0.0, 0.0])
Sigma1 = np.array([[1.0, 0.8], [0.8, 1.5]])
mu2 = np.array([2.0, 2.0])
Sigma2 = np.array([[1.2, -0.5], [-0.5, 0.8]])

# initialization
y = rng.choice([1, 2], size=n, p=[0.2, 0.8])
x_ = np.zeros((n, 2))

# sampling
n1 = np.sum(y == 1); n2 = np.sum(y == 2)
x_[y == 1] = sample_2d_gaussian(mu1, Sigma1, n1, rng)
x_[y == 2] = sample_2d_gaussian(mu2, Sigma2, n2, rng)

# parameters for plotting
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["CMU Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 20
mpl.rcParams["axes.unicode_minus"] = False

# samples plot  
fig, ax = plt.subplots(figsize=(15, 6))

ax.scatter(x_[y==1, 0], x_[y==1, 1], color="blue", label="Component 1")
ax.scatter(x_[y==2, 0], x_[y==2, 1], color="red", label="Component 2")

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.xlim(-5.5,6)
plt.ylim(-3, 4.5)
plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'x_sample.pdf', dpi=1080)
plt.show()

# 80-20 split
x_train, x_test, y_train, y_test = train_test_split(x_, y, test_size=0.2, random_state=29)

# ===== D. Training the Gaussian mixture model =====
# 1. GMM fit
g2 = GaussianMixture(n_components=2, covariance_type='full',
                     reg_covar=1e-6, n_init=10, random_state=29)
g2.fit(x_train)

print("\n1. GMM fit:\n\nFor K=2,")
print("\nWeights:", g2.weights_)
print("\nMeans:", g2.means_)
print("\nCovariances:", g2.covariances_)

# 2. Mapping components to true classes
M = g2.means_
d0 = np.linalg.norm(M[0] - mu1) + np.linalg.norm(M[1] - mu2)
d1 = np.linalg.norm(M[1] - mu1) + np.linalg.norm(M[0] - mu2)
if d0 <= d1:
    class_to_component = {1: 0, 2: 1}
    component_to_class = {0: 1, 1: 2}
else:
    class_to_component = {1: 1, 2: 0}
    component_to_class = {1: 1, 0: 2}

print("\n2. Mapping components to classes:\n\nclass_to_component:", class_to_component)
print("component_to_class:", component_to_class)

# reordering components to match classes
w_hat1 = g2.weights_[class_to_component[1]]
w_hat2 = g2.weights_[class_to_component[2]]
mu_hat1 = g2.means_[class_to_component[1]]
mu_hat2 = g2.means_[class_to_component[2]]
Sigma_hat1 = g2.covariances_[class_to_component[1]]
Sigma_hat2 = g2.covariances_[class_to_component[2]]

print("\nTrue and aligned fitted GMM:")
print("\nTrue weights:", w_true)
print("Aligned fitted weights:", [w_hat1, w_hat2])
print("\nTrue means:\n", np.vstack([mu1, mu2]))
print("\nAligned fitted means:\n", np.vstack([mu_hat1, mu_hat2]))
print("\nTrue covariances:")
print(Sigma1, "\n\n", Sigma2)
print("\nAligned fitted covariances:")
print(Sigma_hat1, "\n\n", Sigma_hat2)

# 3. Verification of the correctness of the GMM model
# L2 norm of errors
w1_err = abs(w_hat1 - w_true[0])
w2_err = abs(w_hat2 - w_true[1])
mu_err1 = np.linalg.norm(mu_hat1 - mu1, ord=2)
mu_err2 = np.linalg.norm(mu_hat2 - mu2, ord=2)
Sig_err1 = np.linalg.norm(Sigma_hat1 - Sigma1, ord=2)
Sig_err2 = np.linalg.norm(Sigma_hat2 - Sigma2, ord=2)

print(f"\n3. Verification of GMM model:")
print(f"\n||w_1(fit) - w_1(true)|| = {w1_err:.4f}")
print(f"||w_2(fit) - w_2(true)|| = {w2_err:.4f}")
print(f"||mu_1(fit) - mu_1(true)||_2 = {mu_err1:.4f}")
print(f"||mu_2(fit) - mu_2(true)||_2 = {mu_err2:.4f}")
print(f"||Sigma_1(fit) - Sigma_1(true)||_2 = {Sig_err1:.4f}")
print(f"||Sigma_2(fit) - Sigma_2(true)||_2 = {Sig_err2:.4f}")

# average log-likelihood on training and test data
train_logl = g2.score(x_train)
test_logl = g2.score(x_test)

print(f"\nAverage log-likelihood per sample (train): {train_logl:.4f}")
print(f"Average log-likelihood per sample (test):  {test_logl:.4f}")

# Draw samples from the fitted GMM (same number as true data)
x_fit, y_fit = g2.sample(n)

# true and fitted GMMs plot
fig, ax = plt.subplots(figsize=(15, 6))

ax.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1],
           color="blue", alpha=0.6, label="True (component 1)")
ax.scatter(x_train[y_train == 2, 0], x_train[y_train == 2, 1],
           color="red", alpha=0.6, label="True (component 2)")

ax.scatter(x_fit[y_fit == 0, 0], x_fit[y_fit == 0, 1], color="orange", alpha=0.7, 
           marker='x', label="Fitted (component 1)")
ax.scatter(x_fit[y_fit == 1, 0], x_fit[y_fit == 1, 1], color="green", alpha=0.7, 
           marker='x', label="Fitted (component 2)")

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.xlim(-5.5,6)
plt.ylim(-3, 4.5)
plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'x_sample_fit.pdf', dpi=1080)
plt.show()

# 1$\sigma$ ellipse
fig, ax = plt.subplots(figsize=(15, 6))

# true ellipses
for k, (mu_true, Sigma_true, color) in enumerate(
        [(mu1, Sigma1, "blue"), (mu2, Sigma2, "red")], start=1):

    # compute eigenvalues & eigenvectors
    eigen_val, eigen_vec = np.linalg.eigh(Sigma_true)

    # sort in descending order
    order = np.argsort(eigen_val)[::-1]
    eigen_val = eigen_val[order]
    eigen_vec = eigen_vec[:, order]

    # rotation angle
    theta = np.degrees(np.arctan2(*eigen_vec[:, 0][::-1])) % 180

    # major and minor axes
    width, height = 2 * np.sqrt(eigen_val)

    # print for reference (optional)
    print(f"\nTrue Component {k}:")
    print(f"  Eigenvalues = {eigen_val}")
    print(f"  Rotation = {theta:.2f} degree")

    ell = Ellipse(xy=mu_true, width=width, height=height, angle=theta, ls="-",
                  edgecolor=color, facecolor='none', lw=3,
                  label=f"True ellipse (component {k})")
    ax.add_patch(ell)

# fitted ellipses
for k, (mu_hat, Sigma_hat, color) in enumerate(
        [(mu_hat1, Sigma_hat1, "green"), (mu_hat2, Sigma_hat2, "orange")], start=1):

    # compute eigenvalues & eigenvectors
    eigen_val, eigen_vec = np.linalg.eigh(Sigma_hat)

    # sort in descending order
    order = np.argsort(eigen_val)[::-1]
    eigen_val = eigen_val[order]
    eigen_vec = eigen_vec[:, order]

    # rotation angle
    theta = np.degrees(np.arctan2(*eigen_vec[:, 0][::-1])) % 180

    # major and minor axes
    width, height = 2 * np.sqrt(eigen_val)

    # print for reference (optional)
    print(f"\nFitted Component {k}:")
    print(f"  Eigenvalues = {eigen_val}")
    print(f"  Rotation = {theta:.2f} degree")

    ell = Ellipse(xy=mu_hat, width=width, height=height, angle=theta, ls="--",
                  edgecolor=color, facecolor='none', lw=4,
                  label=f"Fitted ellipse (component {k})")
    ax.add_patch(ell)

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.xlim(-5.5, 6)
plt.ylim(-3, 4.5)
plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'ellipse.pdf', dpi=1080)
plt.show()

# training and testing data prediction accuracy
comp_pred_train = g2.predict(x_train)
y_pred_train = np.array([component_to_class[c] for c in comp_pred_train])
train_acc = np.mean(y_pred_train == y_train)

comp_pred_test = g2.predict(x_test)
y_pred_test = np.array([component_to_class[c] for c in comp_pred_test])
test_acc = np.mean(y_pred_test == y_test)

print(f"\nTraining accuracy: {train_acc*100:.2f}%")
print(f"Test accuracy: {test_acc*100:.2f}%")

print("\nConfusion matrix (train):")
print(confusion_matrix(y_train, y_pred_train, labels=[1, 2]))

print("\nConfusion matrix (test):")
print(confusion_matrix(y_test, y_pred_test, labels=[1, 2]))


# ===== E. Geometric classification =====
# 1. Visual check for training data and boundary
# geometric parameters
r_hat = (mu_hat2 - mu_hat1) / np.linalg.norm(mu_hat2 - mu_hat1)
m = 0.5 * (mu_hat1 + mu_hat2)

def classify_geom(x):
    return np.where(np.dot(x - m, r_hat) < 0, 1, 2)

# decision boundary
x_vals = np.linspace(np.min(x_train[:,0]) - 1, np.max(x_train[:,0]) + 1, 200)
y_vals = m[1] - (r_hat[0]/r_hat[1]) * (x_vals - m[0])

# plot training boundary and data
fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(x_train[y_train==1,0], x_train[y_train==1,1], c='blue', 
           alpha=0.5, label='Class 1 (train)')
ax.scatter(x_train[y_train==2,0], x_train[y_train==2,1], c='red', 
           alpha=0.5, label='Class 2 (train)')
ax.plot(x_vals, y_vals, 'k--', lw=3, label='Decision boundary')

plt.xlabel(r'$x_1$'); 
plt.ylabel(r'$x_2$')
plt.legend(loc="upper right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'train_bndry.pdf', dpi=1080)
plt.show()

# 2. Visual check for testing data, boundary, and misclassification
# classify testing data
y_geom_test = classify_geom(x_test)

# plot testing decision boundary and data
fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(x_test[y_test==1,0], x_test[y_test==1,1], c='blue', alpha=0.5, 
           label='Class 1 (test)')
ax.scatter(x_test[y_test==2,0], x_test[y_test==2,1], c='red', alpha=0.5, 
           label='Class 2 (test)')
ax.plot(x_vals, y_vals, 'k--', lw=2, label='Decision boundary')

mis_idx = (y_geom_test != y_test)
ax.scatter(x_test[mis_idx,0], x_test[mis_idx,1], facecolors='none', 
           edgecolors='black', lw=1.5, s=80, label='Misclassified')

plt.xlabel(r'$x_1$'); 
plt.ylabel(r'$x_2$')
plt.legend(loc="lower left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'test_bndry.pdf', dpi=1080)
plt.show()

# 3. Confusion matrix (test)
confuse_geom = confusion_matrix(y_test, y_geom_test, labels=[1, 2])
accuracy_geom = accuracy_score(y_test, y_geom_test)
print("\n3. Accuracy and confusion matrix of geometric classifier:\n\nAccuracy (test): {:.2f}%".format(accuracy_geom*100))
print("Confusion matrix (test):")
print(confuse_geom)

# ===== F. Probabilistic classifier with thresholding & ROC =====
# Compute posterior probabilities for each class using the fitted GMM
proba = g2.predict_proba(x_test)

# identify the fitted GMM component corresponding to true class 2
comp_idx_for_class2 = class_to_component[2]

# extract posterior probability
proba_class2 = proba[:, comp_idx_for_class2]

# classification based on threshold
tau = 0.5
y_prob_test = np.where(proba_class2 >= tau, 2, 1)

# 1. Confusion matrix (test)
accuracy_prob = accuracy_score(y_test, y_prob_test)
confuse_prob = confusion_matrix(y_test, y_prob_test, labels=[1, 2])

print(f"\n1. Accuracy and confusion matrix of probabilistic classifier:\n\nAccuracy (test): {accuracy_prob*100:.2f}%")
print("Confusion matrix (test):")
print(confuse_prob)

# 2. ROC curve and AUC
# convert true labels {1,2} to {0,1}
y_test_b = (y_test == 2).astype(int)

# compute ROC curve points across thresholds
f_p, t_p, thresholds = roc_curve(y_test_b, proba_class2)

# compute AUC
roc_auc = auc(f_p, t_p)

print(f"\n2. ROC curve and AUC:\n\nAUC= {roc_auc:.4f}")

# plot ROC curve
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(f_p, t_p, color='darkorange', lw=3, label=f'ROC curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='blue', lw=2, ls='--', label="Chance")

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'roc.pdf', dpi=1080)
plt.show()