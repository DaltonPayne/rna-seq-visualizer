# RNA-Seq Visualizer

This repository contains a Python script for visualizing single-cell RNA sequencing data. It uses Scanpy and various other libraries to preprocess the data, perform dimensionality reduction (PCA and t-SNE), and cluster cells using the Louvain method. Additionally, it generates a t-SNE plot and violin plots for marker genes.

## Dependencies

To use this script, you'll need to have the following Python libraries installed:

- Scanpy
- Numpy
- Pandas
- Matplotlib
- Seaborn
- h5py
- tqdm
- SciPy
- igraph
- python-louvain
- difflib
- argparse

You can install these libraries using pip:

```bash
pip install scanpy numpy pandas matplotlib seaborn h5py tqdm scipy python-igraph louvain difflib argparse
```

## Usage
You can run the script from the command line as follows:

```bash
python main.py -i input_file.h5 -m max_cells -p perplexity -t tsne_output.png -v violin_output.png
```

Replace input_file.h5 with the path to your input data file in H5 format, max_cells with the maximum number of cells to process, perplexity with the perplexity value for t-SNE, and tsne_output.png and violin_output.png with the desired output filenames for the t-SNE and violin plots, respectively.

For example:

```bash
python main.py -i 1M_neurons_filtered_gene_bc_matrices_h5.h5 -m 100000 -p 5 -t tsne_plot.png -v violin_plot.png
```

## License  
This project is licensed under the MIT License. See the LICENSE file for details.