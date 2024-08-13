"""
Created on May 11, 2023

@author: mning
"""
from typing import Callable

import numpy as np
from scipy.fft import rfftfreq

typeFeatNm = list[str]


def genFeatNames(setChn: list[str], thisFeatFn: Callable, params: dict) -> typeFeatNm:
    if (
        thisFeatFn.__name__ == "getMTSpectrogram"
        or thisFeatFn.__name__ == "getMTSpectrogramSbj"
    ):
        featList = ["max", "min", "std", "mean", "spec_kurtosis"]
        band_names = ["delta", "theta", "alpha", "beta"]
        return [
            f"{thisFeatFn.__name__}_{band}_{feat}_{chn}"
            for band in band_names
            for feat in featList
            for chn in setChn
        ]
    elif (
        thisFeatFn.__name__ == "computeFOOOF"
        or thisFeatFn.__name__ == "computeFOOOFCont"
    ):
        if "op_Knee" in params:
            op_Knee = params["op_Knee"]
        if op_Knee:
            apFeat = ["offset", "exponent", "knee"]
        else:
            apFeat = ["offset", "exponent"]
        pFeat = [
            "CF1",
            "PW1",
            "BW1",
            "CF2",
            "PW2",
            "BW2",
            "AUC_alpha",
            "AUC_beta1",
            "AUC_beta2",
        ]
        allFeats = apFeat + pFeat
        return [
            f"{thisFeatFn.__name__}_{feat}_{chn}" for chn in setChn for feat in allFeats
        ]
    elif thisFeatFn.__name__ == "computeFOOOFFreq":
        allFeats = ["AUC_6t12", "AUC_13t30"]
        return [
            f"{thisFeatFn.__name__}_{feat}_{chn}" for chn in setChn for feat in allFeats
        ]
    elif thisFeatFn.__name__ == "computeFOOOF_Alpha":
        apFeat = ["offset", "exponent"]
        pFeat = [
            "CF",
            "PW",
            "BW",
            "AUC_alpha_peak",
            "AUC_alpha_PSD",
            "AUC_peak_8Hz",
            "AUC_PSD_8Hz",
            "AUC_peak_9Hz",
            "AUC_PSD_9Hz",
            "AUC_peak_10Hz",
            "AUC_PSD_10Hz",
            "AUC_peak_11Hz",
            "AUC_PSD_11Hz",
            "AUC_peak_12Hz",
            "AUC_PSD_12Hz",
        ]
        allFeats = apFeat + pFeat
        return [
            f"{thisFeatFn.__name__}_{feat}_{chn}" for chn in setChn for feat in allFeats
        ]
    elif thisFeatFn.__name__ == "getPermutationEntropy":
        if "m" in params:
            m = params["m"]
        # m_str_list = ['m1', 'm2', 'm3', 'm4']
        m_str_list = []
        for i in range(m):
            m_str_list.append(f"m{i}")
        return [f"{chn}_{mstr}" for chn in setChn for mstr in m_str_list]
    elif thisFeatFn.__name__ in ["computePSD", "computePSDCont"]:
        if "Fs" in params:
            Fs = params["Fs"]
        if "NW" in params:
            params["NW"]
        if "n_times" in params:
            n_times = params["n_times"]
        if "total_freq_range" in params:
            total_freq_range = params["total_freq_range"]
            freq_min = total_freq_range[0]
            freq_max = total_freq_range[1]
        else:
            freq_min = 0
            freq_max = np.inf
        freqs = rfftfreq(n_times, 1.0 / Fs)
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        freqs = freqs[freq_mask]
        return [f"{chn}_{freq:.2f}" for chn in setChn for freq in freqs]
    elif (
        thisFeatFn.__name__ == "computeBandpower"
        or thisFeatFn.__name__ == "computeBandpowerSTD"
    ):
        if "binWidth" in params and "endLoop" in params and "bin2_startF" in params:
            binWidth = params["binWidth"]
            endLoop = params["endLoop"]
            bin2_startF = params["bin2_startF"]
            if bin2_startF:
                return [
                    f"{chn}_{(i-1)*binWidth+1}"
                    for chn in setChn
                    for i in range(1, endLoop)
                ]
            else:
                return [
                    f"{chn}_{(i-1)*binWidth}"
                    for chn in setChn
                    for i in range(1, endLoop)
                ]
        else:
            band_names = params["band_names"]
            return [f"{chn}_{band_name}" for chn in setChn for band_name in band_names]
    else:
        return [f"{thisFeatFn.__name__}_{chn}" for chn in setChn]


def genFeatFormatter(
    setChn: list[str], thisFeatFn: Callable, params: dict
) -> typeFeatNm:
    if (
        thisFeatFn.__name__ == "getMTSpectrogram"
        or thisFeatFn.__name__ == "getMTSpectrogramSbj"
    ):
        featList = ["max", "min", "std", "mean", "spec_kurtosis"]
        band_names = ["delta", "theta", "alpha", "beta"]
        return ["%s" for _1 in band_names for _2 in featList for _3 in setChn]
    elif (
        thisFeatFn.__name__ == "computeFOOOF"
        or thisFeatFn.__name__ == "computeFOOOFCont"
    ):
        if "op_Knee" in params:
            op_Knee = params["op_Knee"]
        if op_Knee:
            apFeat = ["offset", "exponent", "knee"]
        else:
            apFeat = ["offset", "exponent"]
        pFeat = [
            "CF1",
            "PW1",
            "BW1",
            "CF2",
            "PW2",
            "BW2",
            "AUC_alpha",
            "AUC_beta1",
            "AUC_beta2",
        ]
        allFeats = apFeat + pFeat
        return ["%s" for _1 in setChn for _2 in allFeats]
    elif thisFeatFn.__name__ == "computeFOOOFFreq":
        allFeats = ["AUC_6t12", "AUC_13t30"]
        return ["%s" for _1 in setChn for _2 in allFeats]
    elif thisFeatFn.__name__ == "computeFOOOF_Alpha":
        apFeat = ["offset", "exponent"]
        pFeat = [
            "CF",
            "PW",
            "BW",
            "AUC_alpha_peak",
            "AUC_alpha_PSD",
            "AUC_peak_8Hz",
            "AUC_PSD_8Hz",
            "AUC_peak_9Hz",
            "AUC_PSD_9Hz",
            "AUC_peak_10Hz",
            "AUC_PSD_10Hz",
            "AUC_peak_11Hz",
            "AUC_PSD_11Hz",
            "AUC_peak_12Hz",
            "AUC_PSD_12Hz",
        ]
        allFeats = apFeat + pFeat
        return ["%s" for _1 in setChn for _2 in allFeats]
    elif thisFeatFn.__name__ == "getPermutationEntropy":
        if "m" in params:
            m = params["m"]
        return ["%s" for _1 in setChn for _2 in range(m)]
    elif thisFeatFn.__name__ in ["computePSD", "computePSDCont"]:
        if "Fs" in params:
            Fs = params["Fs"]
        if "NW" in params:
            params["NW"]
        if "n_times" in params:
            n_times = params["n_times"]
        if "total_freq_range" in params:
            total_freq_range = params["total_freq_range"]
            freq_min = total_freq_range[0]
            freq_max = total_freq_range[1]
        else:
            freq_min = 0
            freq_max = np.inf
        freqs = rfftfreq(n_times, 1.0 / Fs)
        freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
        freqs = freqs[freq_mask]
        return ["%s" for _1 in setChn for _2 in freqs]
    elif (
        thisFeatFn.__name__ == "computeBandpower"
        or thisFeatFn.__name__ == "computeBandpowerSTD"
    ):
        if "endLoop" in params:
            endLoop = params["endLoop"]
            return ["%s" for _1 in setChn for _2 in range(1, endLoop)]
        else:
            band_names = params["band_names"]
            return ["%s" for _1 in setChn for _2 in band_names]
    else:
        return ["%s" for _ in setChn]
