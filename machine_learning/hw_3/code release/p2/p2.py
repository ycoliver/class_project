from os.path import join
from PIL.Image import open
import matplotlib.pyplot as plt
import numpy as np

def preprocess(V):
    print("Data preprocessing")
    min_val = V.min(axis=0)
    V = V - np.asmatrix(np.ones((V.shape[0], 1))) * min_val
    max_val = V.max(axis=0) + 1e-4
    V = (255. * V) / (np.asmatrix(np.ones((V.shape[0], 1))) * max_val) / 100.
    return V

def read_orl(img_size=(112, 92)):
    print("Reading ORL faces database")
    dir = join('../data/ORL_faces', 's')
    V = np.matrix(np.zeros((img_size[0] * img_size[1], 400)))
    for subject in range(40):
        for image in range(10):
            im = open(join(dir + str(subject + 1), str(image + 1) + ".pgm"))
            im = im.resize(img_size[::-1])
            V[:, 10 * subject + image] = np.asmatrix(np.asarray(im).flatten()).T
    return V

def cal_snr(img_origin, img_recon):
    ratio = np.sum(np.power(img_recon, 2)) / np.sum(np.power((img_origin-img_recon), 2))
    return 10 * np.log10(ratio)

img_size = (112, 92)
X = read_orl(img_size)
X_processed = preprocess(X)

print("Performing PCA...")
X_mean = np.mean(X_processed, axis=1)
X_centered = X_processed - X_mean

U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

print(f"Data shape: {X_processed.shape}")
print(f"Number of components: {len(S)}")

fig, axes = plt.subplots(5, 8, figsize=(16, 10))
for i in range(40):
    ax = axes[i // 8, i % 8]
    eigenface = np.asarray(U[:, i]).reshape(*img_size)
    ax.imshow(eigenface, cmap='gray')
    ax.axis('off')
    ax.set_title(f'EF {i+1}', fontsize=10)
plt.suptitle('Top 40 Eigenfaces', fontsize=16, y=0.995)
plt.tight_layout()
plt.savefig('./eigenfaces_top40.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Question 1: Top 40 eigenfaces saved to eigenfaces_top40.pdf")

k_values = [20, 40, 100, 200, 300]
selected_images = [0, 50, 100, 200, 300]

fig, axes = plt.subplots(5, 6, figsize=(18, 15))

for row, img_idx in enumerate(selected_images):
    original = np.asarray(X_processed[:, img_idx]).reshape(*img_size)
    axes[row, 0].imshow(original, cmap='gray')
    axes[row, 0].axis('off')
    if row == 0:
        axes[row, 0].set_title('Original', fontsize=14, fontweight='bold')
    axes[row, 0].set_ylabel(f'Image {img_idx+1}', fontsize=12, rotation=0, ha='right', va='center')
    
    for col, k in enumerate(k_values):
        A_k = U[:, :k]
        theta_k = A_k.T @ X_centered[:, img_idx]
        X_recon = A_k @ theta_k + X_mean
        recon_img = np.asarray(X_recon).reshape(*img_size)
        
        axes[row, col + 1].imshow(recon_img, cmap='gray')
        axes[row, col + 1].axis('off')
        if row == 0:
            axes[row, col + 1].set_title(f'k={k}', fontsize=14, fontweight='bold')

plt.suptitle('Face Reconstruction with Different k Values', fontsize=18, y=0.995)
plt.tight_layout()
plt.savefig('./reconstruction_comparison.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Question 2: Reconstruction comparison saved to reconstruction_comparison.pdf")

k_range = list(range(10, 401, 10))
snr_values = []

print("\nQuestion 3: Calculating SNR for different k values...")
for k in k_range:
    A_k = U[:, :k]
    Theta_k = A_k.T @ X_centered
    X_recon = A_k @ Theta_k + X_mean
    snr = cal_snr(X_processed, X_recon)
    snr_values.append(snr)
    
    if k in [20, 40, 100, 200, 300]:
        print(f"k={k:3d}, SNR={snr:6.2f} dB")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(k_range, snr_values, linewidth=2.5, marker='o', markersize=5, color='#2E8CC9')
ax.set_xlabel('k (Number of Principal Components)', fontsize=14, fontweight='bold')
ax.set_ylabel('SNR (dB)', fontsize=14, fontweight='bold')
ax.set_title('Signal-to-Noise Ratio vs Number of Components', fontsize=16, fontweight='bold')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_linewidth(1.5)
ax.spines['right'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(12)
plt.tight_layout()
plt.savefig('./snr_vs_k.pdf', dpi=300, bbox_inches='tight')
plt.close()
print("Question 3: SNR plot saved to snr_vs_k.pdf")

print("\n=== Variance Analysis ===")
total_variance = np.sum(S**2)
for k in [20, 40, 100, 200, 300]:
    var_explained = np.sum(S[:k]**2) / total_variance * 100
    print(f"Top {k:3d} components explain {var_explained:5.2f}% of variance")

print("\n=== Discussion ===")
print("As k increases:")
print("- Reconstruction quality improves (higher SNR)")
print("- More facial details are preserved")
print("- k=20: Basic face structure visible")
print("- k=40: Major facial features clear")
print("- k=100: Fine details start appearing")
print("- k=200+: Near-perfect reconstruction")
print("This happens because more principal components capture more variance in the data.")

### Final Output
''' 
Question 1: Top 40 eigenfaces saved to eigenfaces_top40.pdf
Question 2: Reconstruction comparison saved to reconstruction_comparison.pdf

Question 3: Calculating SNR for different k values...
k= 20, SNR= 14.53 dB
k= 40, SNR= 16.08 dB
k=100, SNR= 18.93 dB
k=200, SNR= 22.69 dB
k=300, SNR= 27.40 dB
Question 3: SNR plot saved to snr_vs_k.pdf

=== Variance Analysis ===
Top  20 components explain 70.60% of variance
Top  40 components explain 79.23% of variance
Top 100 components explain 89.10% of variance
Top 200 components explain 95.38% of variance
Top 300 components explain 98.43% of variance

=== Discussion ===
As k increases:
- Reconstruction quality improves (higher SNR)
- More facial details are preserved
- k=20: Basic face structure visible
- k=40: Major facial features clear
- k=100: Fine details start appearing
- k=200+: Near-perfect reconstruction
This happens because more principal components capture more variance in the data.
(daixl2) daixunlian@60099741M p2 % 
'''