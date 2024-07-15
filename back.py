import os
import numpy as np
import torch
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import umap
import matplotlib.pyplot as plt

device = torch.device('cuda:0') 

# Load embeddings
A = np.load('data/A.npy')
B = np.load('data/B.npy')
common_component = np.load('data/common_components.npy')

# Print shapes of the arrays for debugging
print("Shape of A:", A.shape)  # (samples, layers, features)
print("Shape of B:", B.shape)  # (samples, layers, features)
print("Shape of common_component:", common_component.shape)  # (samples, layers, features)

# Get the number of layers
num_layers = A.shape[1]

# Define dimension reduction methods
dim_reduction_methods = {
    'PCA': PCA(n_components=2),
    'ICA': FastICA(n_components=2),
    't-SNE': TSNE(n_components=2),
    # 'LDA': LDA(n_components=2),
    'UMAP': umap.UMAP(n_components=2),
    'MDS': MDS(n_components=2)
}

# Iterate over each layer and generate plots
for layer in range(num_layers):
    A_layer = A[:, layer, :]
    B_layer = B[:, layer, :]
    common_component_layer = common_component[:, layer, :]

    # Combine embeddings for each method
    all_embeddings = np.concatenate([A_layer, B_layer, common_component_layer])
    
    for method_name, method in dim_reduction_methods.items():
        if method_name == 'LDA':
            labels_for_lda = np.array([0]*len(A_layer) + [1]*len(B_layer) + [2]*len(common_component_layer))
            reduced_embeddings = method.fit_transform(all_embeddings, labels_for_lda)
        else:
            reduced_embeddings = method.fit_transform(all_embeddings)

        # Print the shape of reduced embeddings for debugging
        print(f"{method_name} reduced_embeddings shape:", reduced_embeddings.shape)

        # Prepare data for plotting
        labels = ['Positive Original', 'Negative Original', 'Common Component']
        colors = ['blue', 'red', 'purple']

        # Create scatter plot
        plt.figure(figsize=(8, 6))
        # Adjusting the scatter plot
        start_index = 0
        for label, color, data in zip(labels, colors, [A_layer, B_layer, common_component_layer]):
            end_index = start_index + len(data)
            plt.scatter(reduced_embeddings[start_index:end_index, 0],
                        reduced_embeddings[start_index:end_index, 1],
                        label=label, color=color)
            start_index = end_index

        plt.legend()
        plt.title(f'{method_name} Visualization of Embeddings - Layer {layer}')
        plt.xlabel(f'{method_name}1')
        plt.ylabel(f'{method_name}2')

        # Create directory if it doesn't exist
        if not os.path.exists(method_name):
            os.makedirs(method_name)

        plt.savefig(f'{method_name}/{method_name.lower()}_visualization_layer_{layer}.png')
        plt.show()
