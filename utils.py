# utils.py
import numpy as np
from scipy.sparse import csr_matrix

def pad_features_if_needed(X_sparse, model):
    required = model.n_features_in_
    if X_sparse.shape[1] < required:
        padded = csr_matrix((X_sparse.shape[0], required))
        padded[:, :X_sparse.shape[1]] = X_sparse
        return padded
    return X_sparse
