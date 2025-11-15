import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
import torch
from torchvision import datasets, transforms

print("Loading MNIST dataset using PyTorch...")
# transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='../data', train=True, download=True)
test_dataset = datasets.MNIST(root='../data', train=False, download=True)

X_train_all = np.array(train_dataset.data.tolist()).reshape(-1, 784) / 255.0
y_train_all = np.array(train_dataset.targets.tolist())
X_test_all = np.array(test_dataset.data.tolist()).reshape(-1, 784) / 255.0
y_test_all = np.array(test_dataset.targets.tolist())

print(f"Full training set: {X_train_all.shape}")
print(f"Full test set: {X_test_all.shape}")

train_indices = []
test_indices = []

for digit in range(10):
    digit_indices = np.where(y_train_all == digit)[0]
    train_indices.extend(digit_indices[:400])
    
for digit in range(10):
    digit_indices = np.where(y_test_all == digit)[0]
    test_indices.extend(digit_indices[:300])

train_indices = np.array(train_indices)
test_indices = np.array(test_indices)

np.random.seed(42)
np.random.shuffle(train_indices)
np.random.shuffle(test_indices)

X_train_full = X_train_all[train_indices]
y_train_full = y_train_all[train_indices]
X_test = X_test_all[test_indices]
y_test = y_test_all[test_indices]

X_train = X_train_full[:3000]
y_train = y_train_full[:3000]
X_val = X_train_full[3000:]
y_val = y_train_full[3000:]

print(f"Training set: {X_train.shape}")
print(f"Validation set: {X_val.shape}")
print(f"Test set: {X_test.shape}")

C_grid = np.logspace(-3, 3, 10)

print("\n" + "="*60)
print("Part (a): Polynomial Kernels")
print("="*60)

for degree in [1, 2]:
    print(f"\n--- Polynomial kernel with degree={degree} ---")
    
    val_errors = []
    
    for C in C_grid:
        clf = svm.SVC(C=C, kernel='poly', degree=degree, gamma='auto')
        clf.fit(X_train, y_train)
        val_error = 1 - clf.score(X_val, y_val)
        val_errors.append(val_error)
        print(f"C={C:.4f}, Validation Error={val_error:.4f}")
    
    best_idx = np.argmin(val_errors)
    best_C = C_grid[best_idx]
    best_val_error = val_errors[best_idx]
    
    print(f"\nBest C: {best_C:.4f}")
    print(f"Best Validation Error: {best_val_error:.4f}")
    
    clf_best = svm.SVC(C=best_C, kernel='poly', degree=degree, gamma='auto')
    clf_best.fit(X_train_full, y_train_full)
    test_error = 1 - clf_best.score(X_test, y_test)
    test_accuracy = clf_best.score(X_test, y_test)
    
    print(f"Test Error: {test_error:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(C_grid, val_errors, linewidth=2.5, marker='o', markersize=8, color='#2E8CC9')
    ax.axvline(x=best_C, color='red', linestyle='--', linewidth=2, label=f'Best C={best_C:.4f}')
    ax.set_xscale('log')
    ax.set_xlabel('C', fontsize=14, fontweight='bold')
    ax.set_ylabel('Validation Error', fontsize=14, fontweight='bold')
    ax.set_title(f'Validation Error vs C (Polynomial Kernel, degree={degree})', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=12)
    ax.spines['top'].set_linewidth(1.5)
    ax.spines['right'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(12)
    plt.tight_layout()
    plt.savefig(f'./poly_degree{degree}_validation_error.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: poly_degree{degree}_validation_error.pdf")

print("\n" + "="*60)
print("Part (b): RBF Kernel")
print("="*60)

gamma_grid = np.logspace(-4, 0, 5)

print(f"\nC grid: {C_grid}")
print(f"Gamma grid: {gamma_grid}")

val_errors_matrix = np.zeros((len(gamma_grid), len(C_grid)))

for i, gamma in enumerate(gamma_grid):
    print(f"\n--- Gamma={gamma:.6f} ---")
    for j, C in enumerate(C_grid):
        clf = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        clf.fit(X_train, y_train)
        val_error = 1 - clf.score(X_val, y_val)
        val_errors_matrix[i, j] = val_error
        print(f"C={C:.4f}, Gamma={gamma:.6f}, Validation Error={val_error:.4f}")

best_idx = np.unravel_index(np.argmin(val_errors_matrix), val_errors_matrix.shape)
best_gamma = gamma_grid[best_idx[0]]
best_C = C_grid[best_idx[1]]
best_val_error = val_errors_matrix[best_idx]

print(f"\nBest C: {best_C:.4f}")
print(f"Best Gamma: {best_gamma:.6f}")
print(f"Best Validation Error: {best_val_error:.4f}")

clf_best = svm.SVC(C=best_C, kernel='rbf', gamma=best_gamma)
clf_best.fit(X_train_full, y_train_full)
test_error = 1 - clf_best.score(X_test, y_test)
test_accuracy = clf_best.score(X_test, y_test)

print(f"Test Error: {test_error:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
for i, gamma in enumerate(gamma_grid):
    ax.plot(C_grid, val_errors_matrix[i, :], linewidth=2.5, marker='o', 
            markersize=8, label=f'γ={gamma:.6f}')
ax.set_xscale('log')
ax.set_xlabel('C', fontsize=14, fontweight='bold')
ax.set_ylabel('Validation Error', fontsize=14, fontweight='bold')
ax.set_title('Validation Error vs C (RBF Kernel)', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=10)
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)
plt.tight_layout()
plt.savefig('./rbf_validation_error.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure saved: rbf_validation_error.pdf")

fig, ax = plt.subplots(figsize=(12, 8))
im = ax.imshow(val_errors_matrix, cmap='viridis', aspect='auto', origin='lower')
ax.set_xticks(range(len(C_grid)))
ax.set_yticks(range(len(gamma_grid)))
ax.set_xticklabels([f'{c:.2e}' for c in C_grid], rotation=45, ha='right')
ax.set_yticklabels([f'{g:.2e}' for g in gamma_grid])
ax.set_xlabel('C', fontsize=14, fontweight='bold')
ax.set_ylabel('Gamma', fontsize=14, fontweight='bold')
ax.set_title('Validation Error Heatmap (RBF Kernel)', fontsize=16, fontweight='bold')
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Validation Error', fontsize=12, fontweight='bold')
ax.plot(best_idx[1], best_idx[0], 'r*', markersize=20, 
        label=f'Best: C={best_C:.4f}, γ={best_gamma:.6f}')
ax.legend(fontsize=12)
plt.tight_layout()
plt.savefig('./rbf_heatmap.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Figure saved: rbf_heatmap.pdf")

print("\n" + "="*60)
print("Summary")
print("="*60)
print("All experiments completed successfully!")
print("Generated files:")
print("- poly_degree1_validation_error.pdf")
print("- poly_degree2_validation_error.pdf")
print("- rbf_validation_error.pdf")
print("- rbf_heatmap.pdf")


### Final Output

'''
============================================================
Part (a): Polynomial Kernels
============================================================

--- Polynomial kernel with degree=1 ---
C=0.0010, Validation Error=0.9100
C=0.0046, Validation Error=0.9100
C=0.0215, Validation Error=0.9100
C=0.1000, Validation Error=0.5780
C=0.4642, Validation Error=0.1680
C=2.1544, Validation Error=0.1020
C=10.0000, Validation Error=0.0880
C=46.4159, Validation Error=0.0790
C=215.4435, Validation Error=0.0900
C=1000.0000, Validation Error=0.0890

Best C: 46.4159
Best Validation Error: 0.0790
Test Error: 0.1097
Test Accuracy: 0.8903
Figure saved: poly_degree1_validation_error.pdf

--- Polynomial kernel with degree=2 ---
C=0.0010, Validation Error=0.9100
C=0.0046, Validation Error=0.9100
C=0.0215, Validation Error=0.9100
C=0.1000, Validation Error=0.9100
C=0.4642, Validation Error=0.7660
C=2.1544, Validation Error=0.2630
C=10.0000, Validation Error=0.1250
C=46.4159, Validation Error=0.0760
C=215.4435, Validation Error=0.0590
C=1000.0000, Validation Error=0.0500

Best C: 1000.0000
Best Validation Error: 0.0500
Test Error: 0.0700
Test Accuracy: 0.9300
Figure saved: poly_degree2_validation_error.pdf

============================================================
Part (b): RBF Kernel
============================================================

C grid: [1.00000000e-03 4.64158883e-03 2.15443469e-02 1.00000000e-01
 4.64158883e-01 2.15443469e+00 1.00000000e+01 4.64158883e+01
 2.15443469e+02 1.00000000e+03]
Gamma grid: [1.e-04 1.e-03 1.e-02 1.e-01 1.e+00]

--- Gamma=0.000100 ---
C=0.0010, Gamma=0.000100, Validation Error=0.9100
C=0.0046, Gamma=0.000100, Validation Error=0.9100
C=0.0215, Gamma=0.000100, Validation Error=0.9100
C=0.1000, Gamma=0.000100, Validation Error=0.9100
C=0.4642, Gamma=0.000100, Validation Error=0.6610
C=2.1544, Gamma=0.000100, Validation Error=0.1870
C=10.0000, Gamma=0.000100, Validation Error=0.1050
C=46.4159, Gamma=0.000100, Validation Error=0.0880
C=215.4435, Gamma=0.000100, Validation Error=0.0800
C=1000.0000, Gamma=0.000100, Validation Error=0.0830

--- Gamma=0.001000 ---
C=0.0010, Gamma=0.001000, Validation Error=0.9100
C=0.0046, Gamma=0.001000, Validation Error=0.9100
C=0.0215, Gamma=0.001000, Validation Error=0.9100
C=0.1000, Gamma=0.001000, Validation Error=0.4270
C=0.4642, Gamma=0.001000, Validation Error=0.1410
C=2.1544, Gamma=0.001000, Validation Error=0.0950
C=10.0000, Gamma=0.001000, Validation Error=0.0800
C=46.4159, Gamma=0.001000, Validation Error=0.0630
C=215.4435, Gamma=0.001000, Validation Error=0.0720
C=1000.0000, Gamma=0.001000, Validation Error=0.0710

--- Gamma=0.010000 ---
C=0.0010, Gamma=0.010000, Validation Error=0.9100
C=0.0046, Gamma=0.010000, Validation Error=0.9100
C=0.0215, Gamma=0.010000, Validation Error=0.3850
C=0.1000, Gamma=0.010000, Validation Error=0.1140
C=0.4642, Gamma=0.010000, Validation Error=0.0700
C=2.1544, Gamma=0.010000, Validation Error=0.0510
C=10.0000, Gamma=0.010000, Validation Error=0.0480
C=46.4159, Gamma=0.010000, Validation Error=0.0480
C=215.4435, Gamma=0.010000, Validation Error=0.0480
C=1000.0000, Gamma=0.010000, Validation Error=0.0480

--- Gamma=0.100000 ---
C=0.0010, Gamma=0.100000, Validation Error=0.9100
C=0.0046, Gamma=0.100000, Validation Error=0.9100
C=0.0215, Gamma=0.100000, Validation Error=0.9100
C=0.1000, Gamma=0.100000, Validation Error=0.8270
C=0.4642, Gamma=0.100000, Validation Error=0.3720
C=2.1544, Gamma=0.100000, Validation Error=0.1560
C=10.0000, Gamma=0.100000, Validation Error=0.1560
C=46.4159, Gamma=0.100000, Validation Error=0.1560
C=215.4435, Gamma=0.100000, Validation Error=0.1560
C=1000.0000, Gamma=0.100000, Validation Error=0.1560

--- Gamma=1.000000 ---
C=0.0010, Gamma=1.000000, Validation Error=0.9100
C=0.0046, Gamma=1.000000, Validation Error=0.9100
C=0.0215, Gamma=1.000000, Validation Error=0.9100
C=0.1000, Gamma=1.000000, Validation Error=0.9100
C=0.4642, Gamma=1.000000, Validation Error=0.9100
C=2.1544, Gamma=1.000000, Validation Error=0.8840
C=10.0000, Gamma=1.000000, Validation Error=0.8840
C=46.4159, Gamma=1.000000, Validation Error=0.8840
C=215.4435, Gamma=1.000000, Validation Error=0.8840
C=1000.0000, Gamma=1.000000, Validation Error=0.8840

Best C: 10.0000
Best Gamma: 0.010000
Best Validation Error: 0.0480
Test Error: 0.0677
Test Accuracy: 0.9323
Figure saved: rbf_validation_error.pdf
Figure saved: rbf_heatmap.pdf

============================================================
Summary
============================================================
All experiments completed successfully!
Generated files:
- poly_degree1_validation_error.pdf
- poly_degree2_validation_error.pdf
- rbf_validation_error.pdf
- rbf_heatmap.pdf
'''
