# Automatic calibration of input shapers
#
# Copyright (C) 2020-2024  Dmitry Butyugin <dmbutyugin@google.com>
#
# This file may be distributed under the terms of the GNU GPLv3 license. 
# From Klipper adapted
import numpy as np
import KOA.shaper_defs
import collections, importlib, logging, math, multiprocessing, traceback
from KOA.simulation import mzv_shaper,zv_shaper,zvd_shaper,PRV

SHAPER_VIBRATION_REDUCTION=20.
DEFAULT_DAMPING_RATIO = 0.1

MIN_FREQ = 5.
MAX_FREQ = 200.
WINDOW_T_SEC = 0.5
MAX_SHAPER_FREQ = 150.

TEST_DAMPING_RATIOS=[0.075, 0.1, 0.15]

AUTOTUNE_SHAPERS = ['zv', 'mzv']

CalibrationResult = collections.namedtuple(
        'CalibrationResult',
        ('name', 'freq', 'vals', 'vibrs', 'smoothing', 'score', 'max_accel'))

def _bisect(func):
        left = right = 1.
        if not func(1e-9):
            return 0.
        while not func(left):
            right = left
            left *= .5
        if right == left:
            while func(right):
                right *= 2.
        while right - left > 1e-8:
            middle = (left + right) * .5
            if func(middle):
                left = middle
            else:
                right = middle
        return left

def find_shaper_max_accel(shaper, scv):
        # Just some empirically chosen value which produces good projections
        # for max_accel without much smoothing
        TARGET_SMOOTHING = 0.12
        max_accel = _bisect(lambda test_accel: get_shaper_smoothing(
            shaper, test_accel, scv) <= TARGET_SMOOTHING)
        return max_accel       

def estimate_shaper(shaper, test_damping_ratio, test_freqs ):

    A, T = shaper.get_impulses()

    inv_D = 1. / A.sum()

    omega = 2. * math.pi * test_freqs
    damping = test_damping_ratio * omega
    omega_d = omega * math.sqrt(1. - test_damping_ratio**2)
    W = A * np.exp(np.outer(-damping, (T[-1] - T)))
    S = W * np.sin(np.outer(omega_d, T))
    C = W * np.cos(np.outer(omega_d, T))
    return np.sqrt(S.sum(axis=1)**2 + C.sum(axis=1)**2) * inv_D

def estimate_remaining_vibrations(shaper, test_damping_ratio,
                                    freq_bins, psd):
    vals = estimate_shaper(shaper, test_damping_ratio, freq_bins)
    # The input shaper can only reduce the amplitude of vibrations by
    # SHAPER_VIBRATION_REDUCTION times, so all vibrations below that
    # threshold can be igonred
    vibr_threshold = psd.max() / SHAPER_VIBRATION_REDUCTION
    remaining_vibrations = np.maximum(
            vals * psd - vibr_threshold, 0).sum()
    all_vibrations = np.maximum(psd - vibr_threshold, 0).sum()
    return (remaining_vibrations / all_vibrations, vals)

def get_shaper_smoothing(shaper, accel=5000, scv=5.):
    half_accel = accel * .5

    A, T = shaper.get_impulses()
    inv_D = 1. / sum(A)
    n = len(T)
    # Calculate input shaper shift
    ts = sum([A[i] * T[i] for i in range(n)]) * inv_D

    # Calculate offset for 90 and 180 degrees turn
    offset_90 = offset_180 = 0.
    for i in range(n):
        if T[i] >= ts:
            # Calculate offset for one of the axes
            offset_90 += A[i] * (scv + half_accel * (T[i]-ts)) * (T[i]-ts)
        offset_180 += A[i] * half_accel * (T[i]-ts)**2
    offset_90 *= inv_D * math.sqrt(2.)
    offset_180 *= inv_D
    return max(offset_90, offset_180)

def fit_shaper(shaper, frequency_data, shaper_freqs=None,max_smoothing=None, scv=5., max_freq=150):

    damping_ratio =  DEFAULT_DAMPING_RATIO
    test_damping_ratios =  TEST_DAMPING_RATIOS

    if not shaper_freqs:
        shaper_freqs = (None, None, None)
    if isinstance(shaper_freqs, tuple):
        freq_end = shaper_freqs[1] or MAX_SHAPER_FREQ
        freq_start = min(shaper_freqs[0] or shaper.min_freq,
                            freq_end - 1e-7)
        freq_step = shaper_freqs[2] or .2
        test_freqs = np.arange(freq_start, freq_end, freq_step)
    else:
        test_freqs = np.array(shaper_freqs)

    max_freq = max(max_freq or MAX_FREQ, test_freqs.max())

    freq_bins = frequency_data[1,:]
    psd = frequency_data[0,freq_bins <= max_freq]
    freq_bins = freq_bins[freq_bins <= max_freq]

    best_res = None
    results = []
    for test_freq in test_freqs[::-1]:
        shaper_vibrations = 0.
        shaper_vals = np.zeros(shape=freq_bins.shape)
        shaper.set_parameters(test_freq, damping_ratio)
        shaper_smoothing = get_shaper_smoothing(shaper, scv=scv)
        if max_smoothing and shaper_smoothing > max_smoothing and best_res:
            return best_res
        # Exact damping ratio of the printer is unknown, pessimizing
        # remaining vibrations over possible damping values
        for dr in test_damping_ratios:
            vibrations, vals = estimate_remaining_vibrations(
                    shaper, dr, freq_bins, psd)
            shaper_vals = np.maximum(shaper_vals, vals)
            if vibrations > shaper_vibrations:
                shaper_vibrations = vibrations
        max_accel = find_shaper_max_accel(shaper, scv)
        # The score trying to minimize vibrations, but also accounting
        # the growth of smoothing. The formula itself does not have any
        # special meaning, it simply shows good results on real user data
        shaper_score = shaper_smoothing * (shaper_vibrations**1.5 +
                                            shaper_vibrations * .2 + .01)
        results.append(
                CalibrationResult(
                    name='MZV', freq=test_freq, vals=shaper_vals,
                    vibrs=shaper_vibrations, smoothing=shaper_smoothing,
                    score=shaper_score, max_accel=max_accel))
        if best_res is None or best_res.vibrs > results[-1].vibrs:
            # The current frequency is better for the shaper.
            best_res = results[-1]
    # Try to find an 'optimal' shapper configuration: the one that is not
    # much worse than the 'best' one, but gives much less smoothing
    selected = best_res
    for res in results[::-1]:
        if res.vibrs < best_res.vibrs * 1.1 and res.score < selected.score:
            selected = res
    return selected


def _ERVA(shaper, frequency_response):
    frequency_data=frequency_response.copy()
    psd=frequency_data[0,:]
    frequencies=frequency_data[1,:]
    zetas=frequency_data[2,0]
    A,T=shaper.get_impulses()
    vals = estimate_shaper(shaper, zetas, frequencies)

    # The input shaper can only reduce the amplitude of vibrations by
    # SHAPER_VIBRATION_REDUCTION times, so all vibrations below that
    # threshold can be igonred
    vibr_threshold = psd.max() / SHAPER_VIBRATION_REDUCTION
    remaining_vibrations = np.maximum(
            vals * psd - vibr_threshold, 0).sum()
    all_vibrations = np.maximum(psd - vibr_threshold, 0).sum()
    return (remaining_vibrations / all_vibrations, vals)


def SSR(shaper, frequency_data,max_smoothing=None, scv=5., max_freq=150):

    damping_ratio =  DEFAULT_DAMPING_RATIO
    test_damping_ratios =  TEST_DAMPING_RATIOS


    freq_bins = frequency_data[1,:]
    psd = frequency_data[0,freq_bins <= max_freq]
    freq_bins = freq_bins[freq_bins <= max_freq]

    best_res = None
    results = []
    shaper_vibrations,vals = _ERVA(shaper,frequency_data)
    shaper_smoothing = get_shaper_smoothing(shaper, scv=scv)
    if max_smoothing and shaper_smoothing > max_smoothing and best_res:
        return best_res
    shaper_score = shaper_smoothing * (shaper_vibrations**1.5 +
                                        shaper_vibrations * .2 + .01)
    return shaper_score

def shaper_score(shaper, frequency_data,max_smoothing=None, scv=5., max_freq=150):

    damping_ratio =  DEFAULT_DAMPING_RATIO
    test_damping_ratios =  TEST_DAMPING_RATIOS


    freq_bins = frequency_data[1,:]
    psd = frequency_data[0,freq_bins <= max_freq]
    freq_bins = freq_bins[freq_bins <= max_freq]

    best_res = None
    results = []
    shaper_vibrations = 0.
    shaper_vals = np.zeros(shape=freq_bins.shape)
    shaper_smoothing = get_shaper_smoothing(shaper, scv=scv)
    if max_smoothing and shaper_smoothing > max_smoothing and best_res:
        return best_res
    # Exact damping ratio of the printer is unknown, pessimizing
    # remaining vibrations over possible damping values
    for dr in test_damping_ratios:
        vibrations, vals = estimate_remaining_vibrations(
                shaper, dr, freq_bins, psd)
        shaper_vals = np.maximum(shaper_vals, vals)
        if vibrations > shaper_vibrations:
            shaper_vibrations = vibrations
    max_accel = find_shaper_max_accel(shaper, scv)
    # The score trying to minimize vibrations, but also accounting
    # the growth of smoothing. The formula itself does not have any
    # special meaning, it simply shows good results on real user data
    shaper_score = shaper_smoothing * (shaper_vibrations**1.5 +
                                        shaper_vibrations * .2 + .01)
    return shaper_score