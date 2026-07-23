import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set paths
thesis_dir = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")
dataset_path = os.path.join(os.path.dirname(thesis_dir), "bist_ai_dataset_real_30cols.csv.xz").replace("\\", "/")
output_image_path = os.path.join(thesis_dir, "images/elbow_method.png").replace("\\", "/")

print(f"Loading dataset from: {dataset_path}")
df_full = pd.read_csv(dataset_path)

# Prepare feature matrix
# Drop target and metadata columns to get features
drop_cols = ['Target', 'Target_T3', 'Target_T5', 'Target_T15', 'Date', 'Ticker', 'Pattern_Type', 
             'Max_Drawdown_15D', 'Max_Gain_15D', 'Timestamp', 'Attack Type']
X = df_full.drop(columns=[col for col in drop_cols if col in df_full.columns], errors='ignore')
X = X.select_dtypes(include=[np.number])

# Remove high-correlation columns to match the project's preprocessing
cor_matrix = X.corr().abs()
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > 0.90)]
X = X.drop(columns=drop_list, errors='ignore')

# Use a representative sample for fast computation of K-Means
X_sample = X.sample(n=5000, random_state=42)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_sample)

# Compute inertias
ks = list(range(1, 11))
inertias = []
for k in ks:
    print(f"Running K-Means for k={k}...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

# Plotting with professional modern style
sns.set_theme(style="white")
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

fig, ax = plt.subplots(figsize=(9, 5.5), dpi=300)

# Plot line and points
ax.plot(ks, inertias, marker='o', markersize=8, color='#1E3A8A', linewidth=2.5, linestyle='-', label='Inertia (Within-Cluster Sum of Squares)')

# Highlight elbow point at k=5
elbow_k = 5
elbow_inertia = inertias[elbow_k - 1]
ax.plot(elbow_k, elbow_inertia, marker='o', markersize=14, markerfacecolor='#EF4444', markeredgecolor='white', markeredgewidth=2, label='Optimal Cluster Number (Elbow Point)')

# Styling grid & spines
ax.grid(True, linestyle='--', alpha=0.5, color='#CBD5E1')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_color('#94A3B8')
ax.spines['bottom'].set_color('#94A3B8')

# Titles and labels
ax.set_title('Elbow Method for Optimal K-Means Clustering', fontsize=15, fontweight='bold', pad=20, color='#0F172A')
ax.set_xlabel('Number of Clusters (k)', fontsize=11, fontweight='semibold', labelpad=10, color='#334155')
ax.set_ylabel('Inertia (distortion)', fontsize=11, fontweight='semibold', labelpad=10, color='#334155')

# Set x-ticks properly
ax.set_xticks(ks)
ax.tick_params(colors='#475569', labelsize=10)

# Add custom annotation for the elbow point
ax.annotate(
    'Elbow Point (k=5)\nInertia drop flattens here',
    xy=(elbow_k, elbow_inertia),
    xytext=(elbow_k + 1.2, elbow_inertia + (max(inertias) - min(inertias)) * 0.1),
    arrowprops=dict(
        arrowstyle="->",
        connectionstyle="arc3,rad=-0.2",
        color='#EF4444',
        lw=2
    ),
    fontsize=10,
    fontweight='bold',
    color='#EF4444',
    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#EF4444", lw=1.5, alpha=0.9)
)

plt.legend(frameon=True, facecolor='white', edgecolor='#E2E8F0', loc='upper right', fontsize=9.5)
plt.tight_layout()

# Save
os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
plt.savefig(output_image_path, dpi=300)
plt.close()

print(f"Modern Elbow Method plot saved successfully to {output_image_path}")
