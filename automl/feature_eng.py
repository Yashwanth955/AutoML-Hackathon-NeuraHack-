from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

def feature_selection(X, y, k=10, method='anova'):
    """Select top k features using specified method"""
    if method == 'anova':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("Invalid feature selection method")
    
    return selector.fit_transform(X, y)

def dimensionality_reduction(X, n_components=0.95):
    """Reduce dimensions using PCA"""
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)