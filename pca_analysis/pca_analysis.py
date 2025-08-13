import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os

# --- Step 1: Load and Prepare Data ---
print("1. Loading and preparing data for PCA...")

# Load the counts table.
counts_df = pd.read_csv('gene_counts.txt', sep='\t', index_col=0, header=0, comment='#')
counts_df = counts_df.drop(columns=['Chr', 'Start', 'End', 'Strand', 'Length'])

# The `featureCounts` output includes summary rows at the bottom. We'll filter these out.
summary_rows = [
    'Unassigned_Unmapped', 'Unassigned_Read_Type', 'Unassigned_Singleton',
    'Unassigned_MappingQuality', 'Unassigned_Chimera', 'Unassigned_FragmentLength',
    'Unassigned_Duplicate', 'Unassigned_MultiMapping', 'Unassigned_Secondary',
    'Unassigned_NonSplit', 'Unassigned_NoFeatures', 'Unassigned_Overlapping_Length',
    'Unassigned_Ambiguity'
]
counts_df = counts_df.loc[~counts_df.index.isin(summary_rows)]

# Clean up column names.
counts_df.columns = [os.path.basename(col).replace('_Aligned.sortedByCoord.out.bam', '') for col in counts_df.columns]

# Load metadata.
metadata_df = pd.read_csv('metadata.csv', index_col=0)

# Transpose the counts table so samples are rows and genes are columns, as required by scikit-learn.
counts_df = counts_df.T

# Ensure sample order matches between counts and metadata.
if not all(counts_df.index == metadata_df.index):
    counts_df = counts_df.loc[metadata_df.index]

# Data normalization is crucial for PCA. We'll use log-transformation and then scale.
# Add a small pseudo-count to avoid log(0).
counts_normalized = np.log2(counts_df + 1)

# Standardize the data so each gene has a mean of 0 and a standard deviation of 1.
# This ensures that genes with high expression don't dominate the PCA.
scaler = StandardScaler()
counts_scaled = scaler.fit_transform(counts_normalized)

print("Data prepared for PCA.")
print("-" * 20)


# --- Step 2: Perform PCA ---
print("2. Performing Principal Component Analysis...")

# Create a PCA object. We only need the first two components for visualization.
pca = PCA(n_components=2)
principal_components = pca.fit_transform(counts_scaled)

# Create a DataFrame for the principal components.
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'], index=metadata_df.index)
pca_df['genotype'] = metadata_df['genotype']

# Calculate the explained variance ratio for each component.
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance of PC1: {explained_variance[0]:.2f}")
print(f"Explained variance of PC2: {explained_variance[1]:.2f}")
print("PCA completed.")
print("-" * 20)


# --- Step 3: Generate and Save the PCA Plot ---
print("3. Generating and saving the PCA plot...")

# Set up the plot.
fig, ax = plt.subplots(figsize=(8, 8))
genotypes = pca_df['genotype'].unique()

# Assign a color and marker for each genotype.
colors = ['red', 'blue']
markers = ['o', 's']

# Plot the data points.
for i, genotype in enumerate(genotypes):
    subset = pca_df[pca_df['genotype'] == genotype]
    ax.scatter(subset['PC1'], subset['PC2'],
               color=colors[i], marker=markers[i], s=100, label=genotype)

# Add labels and a title.
ax.set_xlabel(f"Principal Component 1 ({explained_variance[0]*100:.2f}%)")
ax.set_ylabel(f"Principal Component 2 ({explained_variance[1]*100:.2f}%)")
ax.set_title("PCA of RNA-seq Samples")
ax.legend()
ax.grid(True, linestyle='--', alpha=0.6)

# Save the plot.
plt.savefig('pca_plot.png')
print("PCA plot saved as 'pca_plot.png'.")
print("-" * 20)
print("\nScript finished successfully!")
