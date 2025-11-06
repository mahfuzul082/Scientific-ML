import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import os

# very small constant to avoid division by zero
epsi = 1e-15

# load training data
data = np.genfromtxt("train_hw5.csv", delimiter=",", skip_header=1, dtype=str)

# separate features and labels
x_raw = data[:, :-1].astype(float)     # first 8 columns
labels = data[:, -1]                   # last column

# map A to 0, B to 1
y = np.where(labels == "A", 0, 1)

# unpack columns
E1, px1, py1, pz1, E2, px2, py2, pz2 = [x_raw[:, i] for i in range(8)]

# physics-informed features
E  = E1 + E2 # total energy of the daughters
px = px1 + px2 # total x- momentum
py = py1 + py2 # total y- momentum
pz = pz1 + pz2 # total z- momentum

# total invariant mass
m2 = E**2 - (px**2 + py**2 + pz**2) 
m  = np.sqrt(np.clip(m2, 0, None))

# opening angle
p1 = np.sqrt(px1**2 + py1**2 + pz1**2)
p2 = np.sqrt(px2**2 + py2**2 + pz2**2)
cos_open = (px1*px2 + py1*py2 + pz1*pz2) / np.clip(p1*p2, epsi, None)

# transverse momentum
p_t = np.sqrt(px**2 + py**2)

# energy asymmetry
E_asym = (E1 - E2) / np.clip(E1 + E2, epsi, None)

# feature matrix
x = np.column_stack([m, p_t, cos_open, E_asym])

# 80-20 split and scaling
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, 
                                                    random_state=29, stratify=y)
scaler = StandardScaler()
x_train_s = scaler.fit_transform(x_train)
x_valid_s  = scaler.transform(x_valid)

# logistic regression
clf = LogisticRegression(max_iter=1000, C=2.0)
clf.fit(x_train_s, y_train)

# evaluate validation data
y_pred = clf.predict(x_valid_s)
y_prob = clf.predict_proba(x_valid_s)[:, 1]
acc = accuracy_score(y_valid, y_pred)
cm  = confusion_matrix(y_valid, y_pred)
auc = roc_auc_score(y_valid, y_prob)

print(f"\nAccuracy: {acc*100:.2f}%")
print("\nConfusion matrix:\n", cm)
print(f"\nAUC: {auc:.8f}")

f_p, t_p, _ = roc_curve(y_valid, y_prob)

# ROC plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(f_p, t_p, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc:.8f})')
ax.plot([0, 1], [0, 1], color='blue', lw=2, ls='--', label="Chance")

plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(frameon=False)
plt.tick_params(axis="both", which="both", direction="in")
plt.savefig(f'roc_particle.pdf', dpi=1080)
plt.show()

# predict test data
test_data = np.genfromtxt("test_hw5.csv", delimiter=",", skip_header=1)
E1, px1, py1, pz1, E2, px2, py2, pz2 = [test_data[:, i] for i in range(8)]

E  = E1 + E2; px = px1 + px2; py = py1 + py2; pz = pz1 + pz2
m  = np.sqrt(np.clip(E**2 - (px**2 + py**2 + pz**2), 0, None))
p_t = np.sqrt(px**2 + py**2)
p1 = np.sqrt(px1**2 + py1**2 + pz1**2)
p2 = np.sqrt(px2**2 + py2**2 + pz2**2)
cos_open = (px1*px2 + py1*py2 + pz1*pz2) / np.clip(p1*p2, epsi, None)
E_asym = (E1 - E2) / np.clip(E1 + E2, epsi, None)

x_test = np.column_stack([m, p_t, cos_open, E_asym])
x_test_s = scaler.transform(x_test)

y_test_pred = clf.predict(x_test_s)
    

# probabilistic outputs from the logistic regression
prob = clf.predict_proba(x_test_s)[:, 1]
N_test = len(prob)

# soft counts
N_B_pred_soft = np.sum(prob)
N_A_pred_soft = N_test - N_B_pred_soft

# hard counts
N_B_pred_hard = np.sum(y_test_pred == 1)
N_A_pred_hard = np.sum(y_test_pred == 0)

print("\nPredicted parent counts:")
print(f"\nSoft count:")
print(f"\nN_A_pred = {N_A_pred_soft:.2f}, N_B_pred = {N_B_pred_soft:.2f}")
print(f"\nHard count (threshold 0.5):")
print(f"\nN_A_pred = {N_A_pred_hard}, N_B_pred = {N_B_pred_hard}")
print(f"\nTotal parents in test set: N_test = {N_test}")

y_cd = np.where(y_test_pred == 1, 2, 1).astype(int)

# write labels to text file
if os.path.exists("Mahfuzul_hw5.txt"):
    os.remove("Mahfuzul_hw5.txt")

with open("Mahfuzul_hw5.txt", "w", encoding="utf-8") as f:
    for val in y_cd:
        f.write(f"{val}\n")

print(f"\nSaved Mahfuzul_hw5.txt with {len(y_cd)} predictions.")

# check file
y_check = np.loadtxt("Mahfuzul_hw5.txt", dtype=int)
print(y_check[:10])