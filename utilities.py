import numpy as np
from numpy.linalg import norm

def cosine_similarity(a,b):
    return np.multiply(a,b)/(norm(a)*norm(b))

def cosine_distance(a,X):
    # 0 ~ 2, 0 is nearest
    return 1 - np.true_divide(np.dot(a,X.T),norm(X,axis=1)*norm(a))

def mean_cosine_distance(X,Y):
    return 1 - np.mean(np.true_divide(np.sum(np.multiply(X, Y), axis=1), norm(X, axis=1) * norm(Y, axis=1)))