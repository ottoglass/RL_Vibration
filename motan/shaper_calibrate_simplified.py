import numpy as np
import pandas as pd
from glob import glob
WINDOW_T_SEC = 0.5

MIN_FREQ = 5.
MAX_FREQ = 200.
WINDOW_T_SEC = 0.5
MAX_SHAPER_FREQ = 150.

def split_into_windows( x, window_size, overlap):
    # Memory-efficient algorithm to split an input 'x' into a series
    # of overlapping windows
    step_between_windows = window_size - overlap
    n_windows = (x.shape[-1] - overlap) // step_between_windows
    shape = (window_size, n_windows)
    strides = (x.strides[-1], step_between_windows * x.strides[-1])
    return np.lib.stride_tricks.as_strided(
            x, shape=shape, strides=strides, writeable=False)
    
def psd(x, fs, nfft):
    # Calculate power spectral density (PSD) using Welch's algorithm
    window = np.kaiser(nfft, 6.)
    # Compensation for windowing loss
    scale = 1.0 / (window**2).sum()

    # Split into overlapping windows of size nfft
    overlap = nfft // 2
    x = split_into_windows(x, nfft, overlap)

    # First detrend, then apply windowing function
    x = window[:, None] * (x - np.mean(x, axis=0))

    # Calculate frequency response for each window using FFT
    result = np.fft.rfft(x, n=nfft, axis=0)
    result = np.conjugate(result) * result
    result *= scale / fs
    # For one-sided FFT output the response must be doubled, except
    # the last point for unpaired Nyquist frequency (assuming even nfft)
    # and the 'DC' term (0 Hz)
    result[1:-1,:] *= 2.

    # Welch's algorithm: average response over windows
    psd = result.real.mean(axis=-1)

    # Calculate the frequency bins
    freqs = np.fft.rfftfreq(nfft, 1. / fs)
    return freqs, psd

def calc_freq_response(data):
    N = data.shape[0]
    T = data[-1,0] - data[0,0]
    SAMPLING_FREQ = N / T
    # Round up to the nearest power of 2 for faster FFT
    M = 1 << int(SAMPLING_FREQ * WINDOW_T_SEC - 1).bit_length()
    if N <= M:
        return None

    # Calculate PSD (power spectral density) of vibrations per
    # frequency bins (the same bins for X, Y, and Z)
    fx, px = psd(data[:,1], SAMPLING_FREQ, M)
    fy, py = psd(data[:,2], SAMPLING_FREQ, M)
    fz, pz = psd(data[:,3], SAMPLING_FREQ, M)
    return fx, px+py+pz, px, py, pz


def indicies_of_largest_elements(vector,contribution_percentage):
    sorted_indices = np.argsort(vector)[::-1]
    current_sum = 0
    selected_indices = []
    # Compute the sum of the vector elements
    total_sum = np.sum(vector)
    
    # Compute the target sum
    target_sum = total_sum * contribution_percentage
    for idx in sorted_indices:
        current_sum += vector[idx]
        selected_indices.append(idx)
        if current_sum >= target_sum:
            break
    return selected_indices
    
def freq_from_raw_data(file,contribution_percentage=0.90):
    f,a,_,_,_=calc_freq_response(pd.read_csv(file).to_numpy())
    indicies=indicies_of_largest_elements(a,contribution_percentage)
    amplitudes=a[indicies]
    frequencies=f[indicies]
    zetas=np.random.uniform(low=0.0,high=.4,size=frequencies.size)
    frequency_response=np.vstack((amplitudes,frequencies,zetas))
    return frequency_response