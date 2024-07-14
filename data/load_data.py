import numpy as np

def load_embeddings(file_path):
    return np.load(file_path)

def save_embeddings(embeddings, file_path):
    np.save(file_path, embeddings)
