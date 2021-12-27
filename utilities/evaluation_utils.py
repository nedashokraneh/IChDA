import math
import os
import numpy as np
import pandas as pd

hg19_chr_sizes = {'chr1': 249250621, 'chr2': 243199373, 'chr3': 198022430, 'chr4': 191154276,
 'chr5': 180915260, 'chr6': 171115067, 'chr7': 159138663, 'chr8': 146364022, 'chr9': 141213431,
  'chr10': 135534747, 'chr11': 135006516, 'chr12': 133851895, 'chr13': 115169878, 'chr14': 107349540,
   'chr15': 102531392, 'chr16': 90354753, 'chr17': 81195210, 'chr18': 78077248, 'chr19': 59128983,
    'chr20': 63025520, 'chr21': 48129895, 'chr22': 51304566}


def spatial_correlation(signal,adj_mat):
    dim = len(signal)
    signal_mean = np.mean(signal)
    mat_sum = np.sum(adj_mat)
    frac_nom = 0
    frac_denom = 0
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[0]):
            frac_nom = frac_nom + (adj_mat[i,j] * (signal[i] - signal_mean) * (signal[j] - signal_mean))
        frac_denom = frac_denom + ((signal[i] - signal_mean) * (signal[i] - signal_mean))
    moran = frac_nom * dim / (mat_sum * frac_denom)
    return(moran)

    
def spatial_correlation2(signal1,signal2,adj_mat):
    signal = signal1 + signal2
    signal_mean = np.mean(signal)
    signal_var = np.var(signal)
    mat_sum = np.sum(adj_mat)
    frac_nom = 0
    frac_denom = 0
    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            frac_nom = frac_nom + (adj_mat[i,j] * (signal1[i] - signal_mean) * (signal2[j] - signal_mean))
    moran = frac_nom / (mat_sum * signal_var)
    return(moran)
