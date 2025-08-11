import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import csv

# --- A helper function to parse the GFF3 file for gene annotations ---
def get_annotations_from_gff3(gff3_path):
    """Parses a GFF3 file and returns a DataFrame with gene annotations."""
    annotations = {}
    with open(gff3_path, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            # Skip header lines
            if row[0].startswith('#'):
                continue
            
            # Check if the feature is a 'gene'
            if len(row) > 2 and row[2] == 'gene':
                # The ninth column contains attributes like ID and Name
                attributes = row[8]
                
                gene_id = None
                gene_symbol = None
                
                # Use regex to find the ID and Name
                id_match = re.search(r'ID=(gene:.*?);', attributes)
                name_match = re.search(r'Name=(.*?);', attributes)
                
                if id_match:
                    gene_id = id_match.group(1)
                
                if name_match:
                    gene_symbol = name_match.group(1)
                
                if gene_id:
                    annotations[gene_id] = {'symbol': gene_symbol}
                    
    return pd.DataFrame.from_dict(annotations, orient='index')


# --- Step 1: Load Data and Annotations ---
print("1. Loading data and annotations from GFF3 file...")

# Load the results_df from the previous analysis.
results_df = pd.read_csv('analysis_results.csv', index_col=0)

# Path to your GFF3 file
gff3_path = 'Arabidopsis_thaliana.TAIR10.61.gff3'

# Create the annotation DataFrame by parsing the GFF3 file
annotation_df = get_annotations_from_gff3(gff3_path)

print(f"Results table shape: {results_df.shape}")
print(f"Annotation table shape: {annotation_df.shape}")
print("-" * 20)


# --- Step 2: Filter and Annotate Top Genes ---
print("2. Filtering for top genes and adding annotations...")

# Set cutoffs for significance and fold change
padj_cutoff = 0.05
lfc_cutoff = 1

# Filter for significant genes
significant_genes = results_df[
    (results_df['padj'] < padj_cutoff) &
    (abs(results_df['log2FoldChange']) > lfc_cutoff)
].sort_values('padj')

# Merge the significant genes with the annotation data
annotated_significant_genes = significant_genes.join(annotation_df, how='left')

# Print the top 10 most significant genes with their annotations
print("Top 10 significant genes with annotations:")
print(annotated_significant_genes.head(10)[['symbol', 'log2FoldChange', 'padj']])
print("-" * 20)


# --- Step 3: Visualize the Expression of Top Genes ---
print("3. Generating bar chart for top genes...")

# Select the top 10 most upregulated and top 10 most downregulated genes
top_upregulated = annotated_significant_genes[annotated_significant_genes['log2FoldChange'] > 0].head(10)
top_downregulated = annotated_significant_genes[annotated_significant_genes['log2FoldChange'] < 0].head(10)
top_genes_for_plot = pd.concat([top_upregulated, top_downregulated])

# --- FIX: Create gene_labels by filling NaN values in the 'symbol' column
# with a cleaner version of the gene ID (without the 'gene:' prefix).
# We convert the index to a Series first, which is a valid input for fillna().
gene_labels = top_genes_for_plot['symbol'].fillna(pd.Series(top_genes_for_plot.index).str.replace('gene:', '')).tolist()

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.barh(gene_labels, top_genes_for_plot['log2FoldChange'], color=['red' if fc > 0 else 'blue' for fc in top_genes_for_plot['log2FoldChange']])
ax.set_xlabel('Log2 Fold Change')
ax.set_title('Top 10 Up- and Downregulated Genes (hy5 vs Col-0)')
ax.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('top_genes_bar_chart.png')
print("Bar chart for top genes saved as 'top_genes_bar_chart.png'.")
print("-" * 20)


# --- Step 4: Prepare Genes for Functional Enrichment Analysis ---
print("4. Preparing lists for functional enrichment analysis...")

# Separate genes by direction of change
upregulated_genes_list = annotated_significant_genes[annotated_significant_genes['log2FoldChange'] > 0].index.tolist()
downregulated_genes_list = annotated_significant_genes[annotated_significant_genes['log2FoldChange'] < 0].index.tolist()

# Save the lists to text files
with open('upregulated_genes.txt', 'w') as f:
    for gene in upregulated_genes_list:
        f.write(f"{gene}\n")

with open('downregulated_genes.txt', 'w') as f:
    for gene in downregulated_genes_list:
        f.write(f"{gene}\n")

print(f"Upregulated gene list saved to 'upregulated_genes.txt' ({len(upregulated_genes_list)} genes).")
print(f"Downregulated gene list saved to 'downregulated_genes.txt' ({len(downregulated_genes_list)} genes).")
print("\nScript finished successfully!")
