from nilearn.glm.first_level import spm_hrf
from scipy.signal import fftconvolve
import numpy as np

def apply_hrf_to_embedding(emb, tr=2.0):
    hrf = spm_hrf(tr)
    T, D = emb.shape
    emb_hrf = np.zeros_like(emb)
    for d in range(D):
        emb_hrf[:, d] = fftconvolve(emb[:, d], hrf, mode="full")[:T]
    return emb_hrf