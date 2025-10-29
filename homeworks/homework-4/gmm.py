from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score
from scipy.special import logsumexp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Ellipse

# ===== II. CHAPTER 3: MULTIVARIATE GAUSSIAN MIXTURE MODELING =====
def sample_2d_gaussian(mu, Sigma, n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.multivariate_normal(mean=mu, cov=Sigma, size=n)

# ===== A. STAGE 1: ONE GAUSSIAN: MLE VS. GMM WITH 1 COMPONENT =====
mu = np.array([2, -1])
Sigma = np.array([[2, 0.8], [0.8, 1]])
rng = np.random.default_rng(29)
n = 1000

# ===== 1. Visualization =====
# sampling
x = sample_2d_gaussian(mu, Sigma, n, rng)

np.set_printoptions(precision=16, suppress=False)
print("1.")
print("x: ", x)

# compute eigenvalues and eigenvectors, and sort them
eigen_val, eigen_vec = np.linalg.eigh(Sigma)
order = np.argsort(eigen_val)[::-1] # descending order
eigen_val = eigen_val[order]
eigen_vec = eigen_vec[:, order]

print("\nEigen values: ", eigen_val)
print("\nEigen vectors: ", eigen_vec)

# $1\sigma$ ellipse parameters
wid  = 2*np.sqrt(eigen_val[0]) # major axis
hei = 2*np.sqrt(eigen_val[1]) # minor axis
angle  = np.degrees(np.arctan2(eigen_vec[1,0], eigen_vec[0,0])) % 180 # orientation

print("\nWidth: ", wid)
print("\nHeight: ", hei)
print(f"\nAngle: {angle} degree")

# parameters for plotting
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["CMU Serif"]
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 20
mpl.rcParams["axes.unicode_minus"] = False

# scatter plot (with $1\sigma$ of $\Sigma$)
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x[:,0], x[:,1], color="black", label=r"$\vec{x}$")
ax.add_patch(Ellipse(mu, wid, hei, angle=angle, edgecolor='r', fill=False, lw=3, label=r"True 1$\sigma$ ellipse"))
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"x_sample.pdf", dpi=1080)
plt.show()

# ===== 2. Analytic maximum-likelihood solution ===== 
mu_hat = x.mean(axis=0)
Sigma_hat = ((x - mu_hat).T @ (x - mu_hat)) / len(x)
print("\n2.")
print(f"Maximum likelihood: mu_hat = ", mu_hat)
print(f"\nMaximum likelihood: Sigma_hat = ", Sigma_hat)

# ===== 3. GMM with one component =====
g1 = GaussianMixture(n_components=1, covariance_type='full', n_init=10, random_state=0).fit(x) # one-component fit
mu_gmm = g1.means_[0] # mean
Sigma_gmm = g1.covariances_[0] #covariance
print("\n3.")
print(f"GMM w/ one component: mu_gmm = ", mu_gmm)
print(f"\nGMM w/ one component: Sigma_gmm = ", Sigma_gmm)

mu_err = np.linalg.norm(mu_hat - mu_gmm) # L2 norm
Sigma_err = np.linalg.norm(Sigma_hat - Sigma_gmm, 2) # matrix 2-norm
print(f"\nmu_error:", mu_err)
print(f"\nSigma_error:", Sigma_err)

# ===== 4. Sampling from the fitted model =====
m = 1000
x_new, _ = g1.sample(m)

print("\n4.")
print("x (new): ", x_new)

# compute eigenvalues and eigenvectors, and sort them
eigen_val_new, eigen_vec_new = np.linalg.eigh(Sigma_gmm)
order_new = np.argsort(eigen_val_new)[::-1]
eigen_val_new = eigen_val_new[order_new]
eigen_vec_new = eigen_vec_new[:, order_new]

print("\nEigen values (new): ", eigen_val_new)
print("\nEigen vectors (new): ", eigen_vec_new)

# $1\sigma$ ellipse parameters
wid_new  = 2 * np.sqrt(eigen_val_new[0])
hei_new = 2 * np.sqrt(eigen_val_new[1])
angle_new  = np.degrees(np.arctan2(eigen_vec_new[1,0], eigen_vec_new[0,0])) % 180

print("\nWidth (new): ", wid_new)
print("\nHeight (new): ", hei_new)
print(f"\nAngle (new): {angle_new} degree")

# scatter plot (with $1\sigma$ of $\Sigma$)
fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(x[:,0], x[:,1], color="blue", label=r"$\vec{x}$ (original)")
ax.scatter(x_new[:,0], x_new[:,1], color="red", alpha=0.8, label=r"$\vec{x}$ (new)")
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"x_valid1.pdf", dpi=1080)
plt.show()

# ellipses
fig, ax = plt.subplots(figsize=(12, 6))
ax.add_patch(Ellipse(mu, wid, hei, angle=angle, edgecolor="blue", fill=False, lw=3, label=r"True 1$\sigma$ ellipse (original)"))
ax.add_patch(Ellipse(mu_gmm, wid_new, hei_new, linestyle="--", angle=angle, edgecolor="red", fill=False, lw=3, label=r"True 1$\sigma$ ellipse (new)"))
ax.set_xlim(mu[0] - 4, mu[0] + 4)
ax.set_ylim(mu[1] - 4, mu[1] + 4)
plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"x_valid2.pdf", dpi=1080)
plt.show()

# ===== STAGE 2: TWO-COMPONENT MIXTURE IN 2D: FIT, COMPARE, SAMPLE, CLASSIFY =====
w_true = np.array([0.2, 0.8])
mu1 = np.array([0.0, 0.0])
Sigma1 = np.array([[1.0, 0.6], [0.6, 1.5]])
mu2 = np.array([4.0, 3.0])
Sigma2 = np.array([[1.2, -0.5], [-0.5, 0.8]])

z = rng.choice([0, 1], size=n, p=[0.2, 0.8])
x_ = np.zeros((n, 2))

# ===== 1. Visualization =====
# sampling
for k in [0, 1]:
    n_ = np.sum(z==k)
    x_[z==k] = sample_2d_gaussian([mu1, mu2][k], [Sigma1, Sigma2][k], n_, rng)

# scatter plot    
fig, ax = plt.subplots(figsize=(15, 6))

ax.scatter(x_[z==0, 0], x_[z==0, 1], color='blue', label='Type 1')
ax.scatter(x_[z==1, 0], x_[z == 1, 1], c='red', label='Type 2')

plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.legend(loc="upper left", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'x_sample2.pdf', dpi=1080)
plt.show()

# ===== 2. Modeling =====
# fit GMMs
g1 = GaussianMixture(n_components=1, covariance_type='full', random_state=42)
g1.fit(x_)
print("\n2. ")
print("For K=1,\nMean:", g1.means_)
print("\nCovariance:", g1.covariances_)

g2 = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
g2.fit(x_)
print("\nFor K=2,\nMean:", g2.means_)
print("\nCovariance:", g2.covariances_)

# compute diagnostics
logL1, bic1 = g1.score(x_) * len(x_), g1.bic(x_)
logL2, bic2 = g2.score(x_) * len(x_), g2.bic(x_)

print(f"\nK=1: log-likelihood = {logL1:.16f}, Bayesian information criterion = {bic1:.16f}")
print(f"\nK=2: log-likelihood = {logL2:.16f}, Bayesian information criterion = {bic2:.16f}")

# ===== 3. Comparisons =====
means2, covs2, weights2 = g2.means_, g2.covariances_, g2.weights_
print("\n3. ")
print("For K = 2, ")
print(f"\nWeights: ", weights2)
print(f"\nMeans: ", means2)
print(f"\nCovariances: ", covs2)

# ===== Sampling from the fitted model =====
x1_new, z_new = g2.sample(m)

# scatter plot
fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(x_[:,0], x_[:,1], color="black", alpha=0.35, label="True data samples")
ax.scatter(x1_new[z_new==0, 0], x1_new[z_new==0, 1], color="blue", alpha=0.5, label="New samples (Type 1)")
ax.scatter(x1_new[z_new==1, 0], x1_new[z_new==1, 1], c="red", alpha=0.5, label="New samples (Type 2)")

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend(loc="lower right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"x_sample_from_fit.pdf", dpi=1080)
plt.show()

# ===== 5. Classification =====
# setup
n_test = 1000
rng = np.random.default_rng(129)
z_true = rng.choice([0, 1], size=n_test, p=w_true)
x_test = np.zeros((n_test, 2))

# constructing ground truth by sampling and testing the GMM model
for k in [0, 1]:
    n_ = np.sum(z_true==k)
    x_test[z_true==k] = sample_2d_gaussian([mu1, mu2][k], [Sigma1, Sigma2][k], n_, rng)
    
predicted_labels = g2.predict(x_test)

print("\n5. ")
#print(f"Predicted labels: ", predicted_labels)

# plots
fig, ax = plt.subplots(figsize=(15, 6))
ax.scatter(x_test[predicted_labels==0,0], x_test[predicted_labels==0,1], color="blue",
           alpha=0.7, label="Type 1 (predicted)")
ax.scatter(x_test[predicted_labels==1,0], x_test[predicted_labels==1,1], color="red",
           alpha=0.7, label="Type 2 (predicted)")

plt.xlabel(r"$x_1$")
plt.ylabel(r"$x_2$")
plt.legend(loc="lower right", frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f"class.pdf", dpi=1080)
plt.show()

# compute accuracy and build confusion matrix
acc = accuracy_score(z_true, predicted_labels)
cm  = confusion_matrix(z_true, predicted_labels)

print(f"\nClassification accuracy: {acc:.3f}")
print("\nConfusion matrix:\n", cm)