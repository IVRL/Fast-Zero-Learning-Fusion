import numpy as np

import cv2
from scipy.ndimage.filters import convolve
from scipy.spatial.distance import cdist
from cv2.ximgproc import guidedFilter

def Guided(I, p, r, eps):
    return guidedFilter(I, p, r, eps)

def GuidedOptimize(G, P, r, eps):
    N = len(G)
    W = []
    
    for i in range(N):
        # MOST COSTLY OPERATION IN THE WHOLE THING
        W.append(Guided(G[i].astype(np.float32), P[i].astype(np.float32), r, eps))
    
    W = np.dstack(W) + 1e-12
    W = W / W.sum(axis=2, keepdims=True)
    return W

def SalWeights(G):
    N = len(G)
    W = []
    
    for i in range(N):
        W.append(saliency(G[i]))

    W = np.dstack(W) + 1e-12
    W = W / W.sum(axis=2, keepdims=True)
    return W

D = cdist(np.arange(256)[:,None], np.arange(256)[:,None], 'euclidean')
def saliency(img):
    global D
    hist = np.bincount(img.flatten(), minlength=256) / img.size
    sal_tab = np.dot(hist, D)
    z = sal_tab[img]
    return z

def FuseWeights(G, W):
    return np.sum(np.dstack(G)*W, axis=2)