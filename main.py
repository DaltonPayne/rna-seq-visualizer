import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from tqdm import tqdm
import scipy.sparse
import random
import igraph
import louvain
import difflib
import argparse

def print_structure(file_path):
    with h5py.File(file_path, 'r') as f:
        def print_attrs(name, obj):
            print(name)
            for key, val in obj.attrs.items():
                print("  {}: {}".format(key, val))

        f.visititems(print_attrs)


file_path = '1M_neurons_filtered_gene_bc_matrices_h5.h5'
print_structure(file_path)

def filter_genes_cells(adata, markers):
    matched_markers = [difflib.get_close_matches(marker, adata.var_names, n=1, cutoff=0.8) for marker in markers]
    matched_markers = [match[0] for match in matched_markers if len(match) > 0]

    if not matched_markers:
        raise ValueError("No matching marker genes found. Please check the provided marker gene names.")

    marker_expression = adata[:, matched_markers].to_df()
    marker_sum = marker_expression.sum(axis=1)
    genes_cells = marker_sum > 0
    adata_filtered = adata[genes_cells, :]

    return adata_filtered

def make_unique(names, separator='_'):
    counts = {}
    unique_names = []
    for name in names:
        if name in counts:
            counts[name] += 1
            unique_name = f"{name}{separator}{counts[name]}"
        else:
            counts[name] = 0
            unique_name = name
        unique_names.append(unique_name)
    return unique_names

def read_data(file_path, max_cells=None):
    with h5py.File(file_path, 'r') as f:
        gene_bc_group = f['mm10']
        data = gene_bc_group['data']
        indices = gene_bc_group['indices']
        indptr = gene_bc_group['indptr']
        shape = gene_bc_group['shape']

        if max_cells is not None:
            all_cells = list(range(shape[1]))
            cell_indices = sorted(random.sample(all_cells, max_cells))
        else:
            cell_indices = list(range(shape[1]))

        data_list, indices_list, indptr_list = [], [], [0]

        for cell_idx in tqdm(cell_indices, desc='Reading cells'):
            start, end = indptr[cell_idx], indptr[cell_idx + 1]
            data_list.extend(data[start:end])
            indices_list.extend(indices[start:end])
            indptr_list.append(len(data_list))

        gene_names_ds = f['mm10/gene_names']
        gene_names = [name.decode('utf-8') for name in gene_names_ds]
        
        adata = sc.AnnData(scipy.sparse.csr_matrix((data_list, indices_list, indptr_list),
                                                   shape=(len(cell_indices), shape[0])))

        adata.obs_names = gene_bc_group['barcodes'][cell_indices].astype(str)
        gene_names = gene_bc_group['gene_names'][:].astype(str)
        adata.var_names = make_unique(gene_names)
        
    return adata

def preprocess_data(adata, min_genes=200, min_cells=3):
    print("Preprocessing data...")
    sc.pp.filter_cells(adata, min_genes=min_genes)
    sc.pp.filter_genes(adata, min_cells=min_cells)
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    return adata

def perform_pca(adata, n_comps=None): 
    print("Performing PCA...")
    if n_comps is None:
        n_comps = min(adata.shape) - 1
    sc.tl.pca(adata, svd_solver='arpack', n_comps=n_comps)
    return adata


def perform_tsne(adata, perplexity=30):
    print("Performing t-SNE...")
    sc.tl.tsne(adata, perplexity=perplexity)
    return adata

def perform_clustering(adata):
    print("Performing Louvain clustering...")
    actual_n_pcs = adata.obsm['X_pca'].shape[1]
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=actual_n_pcs)
    sc.tl.louvain(adata)
    
    # Find the most highly expressed gene for each cluster
    cluster_genes = []
    for cluster in np.unique(adata.obs['louvain']):
        cluster_data = adata[adata.obs['louvain'] == cluster]
        mean_expression = cluster_data.X.mean(axis=0)
        top_gene_idx = np.argmax(mean_expression)
        top_gene_name = adata.var_names[top_gene_idx]
        cluster_genes.append(top_gene_name)

    adata.obs['top_gene'] = adata.obs['louvain'].map(dict(zip(np.unique(adata.obs['louvain']), cluster_genes)))

    return adata


def main():
    parser = argparse.ArgumentParser(description='RNA-Seq Visualizer')
    parser.add_argument('-i', '--input', type=str, default='1M_neurons_filtered_gene_bc_matrices_h5.h5', help='Input data file (H5 format)')
    parser.add_argument('-m', '--max_cells', type=int, default=100000, help='Maximum number of cells to process')
    parser.add_argument('-p', '--perplexity', type=int, default=5, help='Perplexity for t-SNE')
    parser.add_argument('-t', '--tsne_output', type=str, default='tsne_plot.png', help='Output file for t-SNE plot')
    parser.add_argument('-v', '--violin_output', type=str, default='violin_plot.png', help='Output file for violin plot')

    args = parser.parse_args()

    file_path = args.input
    max_cells = args.max_cells
    perplexity = args.perplexity
    tsne_save_path = args.tsne_output
    violin_save_path = args.violin_output
    
    adata = read_data(file_path, max_cells=max_cells)

    print("Gene names in the dataset:")
    for gene_name in adata.var_names:
        print(gene_name)

    neuron_markers = ['Smad2', 'Smad4', 'Smad7']

    adata_genes = filter_genes_cells(adata, neuron_markers)
    adata_genes.var['gene_name'] = adata_genes.var_names
    adata_genes = preprocess_data(adata_genes)

    adata_genes = perform_tsne(adata_genes, perplexity=perplexity)
    adata_genes = perform_clustering(adata_genes)

    available_neuron_markers = [gene for gene in neuron_markers if gene in adata_genes.var_names]

    if available_neuron_markers:
        print("Generating violin plot...")
        sc.pl.violin(adata_genes, available_neuron_markers, groupby='louvain', rotation=90, gene_symbols='gene_name',
        title='Violin plot of selected genes', save=violin_save_path)
        print("Violin plot generated.")
    else:
        print("None of the specified neuron markers are available in the dataset.")

    adata_genes = perform_tsne(adata_genes, perplexity=perplexity)
    adata_genes = perform_clustering(adata_genes)

    # Generating tsne plot
    print("Generating tsne plot...")
    sc.pl.tsne(adata_genes, color='top_gene', title='t-SNE Visualization of Gene Expression Patterns in Single Cells', save=tsne_save_path)
    print("tsne plot generated.")
    
if __name__ == '__main__':
    main()