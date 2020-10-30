from sklearn.decomposition import PCA, FastICA
from sklearn.preprocessing import minmax_scale, scale, normalize

def preprocess_decompose(X, n_components, strategy="PCA"):
    if strategy == "PCA":
        decomposition = PCA(n_components=n_components)
    elif strategy == "ICA":
        decomposition = FastICA(n_components=n_components)
    else:
        raise Exception("Strategy must be 'PCA' or 'ICA'")
    X_decomp = decomposition.fit_transform(X)
    return X_decomp

def preprocess_data(X, strategy="std", axis=0):
    if strategy == "std":
        X_feat = scale(X, axis=axis)
    elif strategy == "minmax":
        X_feat = minmax_scale(X, axis=axis)
    elif strategy == "norm":
        X_feat = normalize(X, axis=axis)
    else:
        raise Exception("Strategy must be 'std', 'minmax' or 'norm'")
    return X_feat

def preprocess_features(X, strategy="std"):
    return preprocess_data(X, strategy, 0)

def preprocess_vectors(X, strategy="std"):
    return preprocess_data(X, strategy, 1)