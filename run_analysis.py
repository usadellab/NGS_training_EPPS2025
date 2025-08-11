import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats

# --- Step 1: Load Data ---
print("1. Loading data...")

# Load the counts table with the correct header and a simple comment filter.
# The 'comment='#' will skip the first line.
# 'header=0' tells pandas to use the second line of the file (index 0 after skipping the comment) as the column names.
counts_df = pd.read_csv('gene_counts.txt', sep='\t', index_col=0, header=0, comment='#')

# Remove the extra columns that are not count data.
# The `drop` function is a reliable way to remove these columns by name.
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

# --- FIX: Clean up the column names from featureCounts output ---
# The columns are the full paths to the BAM files. We need to extract just the sample name.
counts_df.columns = [os.path.basename(col).replace('_Aligned.sortedByCoord.out.bam', '') for col in counts_df.columns]

# Load the metadata file.
metadata_df = pd.read_csv('metadata.csv', index_col=0)

# --- FIX: Transpose the counts table to have samples as rows and genes as columns ---
# This is a critical step for compatibility with pydeseq2.
counts_df = counts_df.T

# Check that the sample order matches between counts and metadata.
# This is a critical check to avoid errors.
if not all(counts_df.index == metadata_df.index):
    # If they don't match, reorder the counts rows to match the metadata.
    counts_df = counts_df.loc[metadata_df.index]

# Display a quick check of the dataframes
print(f"Counts table shape: {counts_df.shape}")
print(f"Metadata table shape: {metadata_df.shape}")
print("-" * 20)


# --- Step 2: Run PyDESeq2 Differential Expression Analysis ---
print("2. Running PyDESeq2 analysis...")

# Create the DeseqDataSet object.
# The `design='~genotype'` parameter tells the model to compare gene expression
# based on the 'genotype' column in the metadata.
dds = DeseqDataSet(
    counts=counts_df,
    metadata=metadata_df,
    design='~genotype',
    refit_cooks=True # Recommended for robustness
)

# Run the full pipeline
dds.deseq2()
print("PyDESeq2 pipeline completed.")
print("-" * 20)


# --- Step 3: Extract and Filter Results ---
print("3. Extracting and filtering results...")

# Get the results for the contrast 'hy5' vs 'Col-0'.
# The `contrast` list specifies the comparison: [variable, test_level, reference_level].
stat_results = DeseqStats(dds, contrast=['genotype', 'hy5', 'Col-0'])
stat_results.run_wald_test()
stat_results.summary() # <-- ADDED: Call the summary method to build the results table.
results_df = stat_results.results_df

# Filter for significant genes based on adjusted p-value (padj) and log2 fold change.
padj_cutoff = 0.05
lfc_cutoff = 1

significant_genes = results_df[
    (results_df['padj'] < padj_cutoff) &
    (abs(results_df['log2FoldChange']) > lfc_cutoff)
]

print(f"Found {len(significant_genes)} significant genes.")
print("-" * 20)


# --- Step 4: Generate and Save Volcano Plot ---
print("4. Generating and saving volcano plot...")

# Create a figure and axis for the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Define colors for different gene states
colors = []
for index, row in results_df.iterrows():
    if row['padj'] < padj_cutoff and row['log2FoldChange'] > lfc_cutoff:
        colors.append('red') # Upregulated
    elif row['padj'] < padj_cutoff and row['log2FoldChange'] < -lfc_cutoff:
        colors.append('blue') # Downregulated
    else:
        colors.append('gray') # Not significant

# Scatter plot of all genes
ax.scatter(results_df['log2FoldChange'], -np.log10(results_df['padj']), c=colors, alpha=0.6, s=15)

# Add labels and title
ax.set_xlabel("Log2 Fold Change (hy5 vs Col-0)")
ax.set_ylabel("-log10(Adjusted p-value)")
ax.set_title("Volcano Plot for PyDESeq2 Analysis")
ax.grid(True, linestyle='--', alpha=0.5)

# Add lines for the cutoffs
ax.axhline(-np.log10(padj_cutoff), color='black', linestyle='--')
ax.axvline(lfc_cutoff, color='black', linestyle='--')
ax.axvline(-lfc_cutoff, color='black', linestyle='--')

# Save the figure
plt.savefig('volcano_plot.png')
print("Volcano plot saved as 'volcano_plot.png'.")
print("-" * 20)


# --- Step 5: Final Check ---
# Check the results for the gene mentioned in the paper.
# Correct the gene ID format to match the featureCounts output.
gene_of_interest = 'gene:AT5G11260' 
if gene_of_interest in results_df.index:
    print(f"Analysis for gene '{gene_of_interest}':")
    print(results_df.loc[gene_of_interest])
else:
    print(f"Gene ID '{gene_of_interest}' not found in the results.")

print("\nScript finished successfully!")
