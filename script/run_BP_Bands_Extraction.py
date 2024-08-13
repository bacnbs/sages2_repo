"""
Created on Sep 26, 2023

Compute bandpower from PSD

@author: mning
"""
import logging
import os
import time
from types import TracebackType
from typing import Any
from typing import Dict

import matplotlib.pyplot as plt
import mne
import numpy as np

plt.set_loglevel(level="warning")

# import polars as pl
# from joblib import Parallel, delayed

import sys

from EEGFeaturesExtraction.coreSrc import (
    getAllFiles,
    getFeatFnMd,
    genFeatMeta,
    getDuke1020Md,
)
import EEGFeaturesExtraction.utils.log as log
from EEGFeaturesExtraction.utils import getChns, customReference

# import coreSrc.getAllFiles as getAllFiles
# import coreSrc.getFeatFnMd as getFeatFnMd
# import coreSrc.genFeatMeta as genFeatMeta
# import utils.log as log

# from .coreSrc import getAllFiles
# import getFeatFnMd
# import log
# import genFeatMeta
# import LyE_R

op_SAGES = 0
testCase = 0
plotOp = 0
is_EO = 1
# binWidth = 1
# endLoop = 30
op_Knee = 0
op_mV = 1
# 'bipolar', 'eqc' short for equal gain combining, 'car' short for common average re-referencing
op_Reref = "car"
if op_Knee:
    knee_str = "knee"
else:
    knee_str = "fixed"
# relativeFOp is for getSpecStatsMd.bandpower
relativeFOp = True
if relativeFOp:
    relative_str = "Relative"
else:
    relative_str = "Absolute"

psd_estimation = "multitaper"
op_flatten = 1

paramOps: Dict[str, Any] = dict()
paramOps["testCase"] = testCase
paramOps["plotOp"] = plotOp
paramOps["relativeFOp"] = relativeFOp

if is_EO:
    eo_str = "EO"
    if op_SAGES:
        matchStr = "_EO_S06_59.set"
    else:
        startStr = "INT"
        matchStr = "_EO_S06_26.set"
else:
    eo_str = "EC"
    if op_SAGES:
        matchStr = "_EC_S06_59.set"
    else:
        startStr = "INT"
        matchStr = "_EC_S06_26.set"

if op_mV:
    unit_str = "mV"
else:
    unit_str = "V"

currDir = os.getcwd()
print(currDir)

logger = log.setup_custom_logger("root", logging.DEBUG)
# logger = log.setup_custom_logger('root', logging.NOTSET)

if op_SAGES:
    dirNamePrep = r"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\DATA_preprocessed_MN_Copy"
    dirNameOutput = r"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\FeatureStores"
    dirFigOutput = r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data\Baseline_rsEEG_preprocessing\DATA_visualization"
else:
    dirNamePrep = r"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA\DATA_Preprocessed"
    dirNameOutput = r"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA\FeatureStores"
    dirFigOutput = (
        r"C:\Users\mning\Documents\CNBS_Projects\Duke_Preop_EEG\DATA\DATA_visualization"
    )

if testCase:
    featStoreFN = r"featureStore_Sbj_test.csv"
else:
    if op_mV:
        featStoreFN = rf"{dirNameOutput}\featureStore_Sbj_BP_Bands_{eo_str}_{relative_str}_car_{unit_str}.csv"
    else:
        featStoreFN = rf"{dirNameOutput}\featureStore_Sbj_BP_Bands_{eo_str}_{relative_str}_car.csv"

featList = getFeatFnMd.getBPFnSbj()

# [sbj x feature [x trials]]
# featureStore is a 2D array with sbj ID and trial # as identifier column stored in excel spreadsheet
# 3rd column indicate presence of delirium and the rest of the columns are feature names
# featureStore = np.empty((1,numFeats))

if op_SAGES:
    fileList, sbjList = getAllFiles.getAllFilesEndingRegex(dirNamePrep, matchStr)
else:
    fileList, sbjList = getAllFiles.getAllFilesBeginEndingRegex(
        dirNamePrep, startStr, matchStr
    )
    duke1020 = getDuke1020Md.getDuke1020()
    fileList = [
        fileName
        for iFile, fileName in enumerate(fileList)
        if sbjList[iFile] in duke1020
    ]
    sbjList = [sbj for sbj in sbjList if sbj in duke1020]

if testCase:
    fileList = fileList[2:4]
    sbjList = sbjList[2:4]

feature_names = ["SbjID", "TrialNum", "SetFile"]
feature_formatter = ["%s", "%s", "%s"]

# a priori featured freq range
freqBands = [[1, 4], [4, 8], [8, 12], [12, 20]]
band_names = ["delta", "theta", "alpha", "beta"]
# freqBands = []
# band_names = []
# for i in range(1, endLoop):
#     #freqBands.append([i, i+1])
#     freqBands.append([(i-1)*binWidth+1, i*binWidth+1])
#     band_names.append([str((i-1)*binWidth+1) + 'Hz'])

# params defined here.
# TODO: create json schema for specification
# TODO: then load json file into params dict variable
# TODO: log param into file. Also log git commit number, date, username, etc.
params: Dict[str, Any] = dict()
params["NW"] = 2
params["window_length"] = int(round(1 * 500))
params["window_step"] = int(round(0.5 * 500))
params["band_freq"] = freqBands
params["total_freq_range"] = [1, 90.0]
params["band_names"] = band_names
params["paramOps"] = paramOps
params["op_Knee"] = op_Knee
params["op_Reref"] = op_Reref
params["is_EO"] = is_EO
params["psd_estimation"] = psd_estimation
params["op_flatten"] = op_flatten
params["n_times"] = 1500
# params['binWidth'] = binWidth
# params['endLoop'] = endLoop
params["op_mV"] = op_mV

# or find peak freq in alpha range

if op_Reref == "bipolar" or op_Reref == "egc":
    anode, cathode, ch_dropped = getChns.getAnodeCathode()

featureStore = None

for iterCnt in range(len(fileList)):
    # remove try and catch for easier debugging and catching bugs
    try:
        tic = time.perf_counter()
        # shape = [#trials, #channel, #sample points]
        # 59 channels, 1500 sample points
        setInfo = mne.io.read_epochs_eeglab(fileList[iterCnt])
        if op_Reref == "bipolar":
            mne.set_bipolar_reference(setInfo, anode=anode, cathode=cathode, copy=False)
            # setInfo.drop_channels(ch_dropped, on_missing='raise')
            if logger.isEnabledFor(logging.DEBUG) and iterCnt == 0:
                thisMontage = setInfo.get_montage()
                thisMontage.plot()
        elif op_Reref == "egc":
            customReference.set_egc_reference(
                setInfo, anode=anode, cathode=cathode, copy=False
            )
            # setInfo.drop_channels(ch_dropped, on_missing='raise')
            if logger.isEnabledFor(logging.DEBUG) and iterCnt == 0:
                thisMontage = setInfo.get_montage()
                thisMontage.plot()
        params["Fs"] = setInfo.info["sfreq"]
        params["epochLen"] = 3 * setInfo.info["sfreq"]
        # in V as per https://mne.tools/stable/overview/implementation.html#internal-representation-units
        setDat = setInfo.get_data()
        setChn = setInfo.info["ch_names"]
        params["chnNames"] = setChn
        if iterCnt == 0:
            params["ref_chn"] = setChn
        params["saveDir"] = os.path.join(dirFigOutput, sbjList[iterCnt])
        if not os.path.exists(os.path.join(dirFigOutput, sbjList[iterCnt])):
            os.makedirs(os.path.join(dirFigOutput, sbjList[iterCnt]))

        numTrials = np.size(setDat, 0)

        # For now use list, convert to polars later
        # featBlock.dtype.type is numpy.str_
        featBlock = np.array([sbjList[iterCnt], 0, fileList[iterCnt]]).reshape(-1, 1).T

        logging.debug(f"{np.shape(featBlock)=}")

        # [features [x trials]]
        # 1st (activity), 2nd (mobility) & 3rd (complexity) Hjorth parameters
        for thisFeatFn in featList:
            featBlock = np.hstack((featBlock, thisFeatFn(setDat, params)))
            if iterCnt == 0:
                feature_names += genFeatMeta.genFeatNames(setChn, thisFeatFn, params)
                feature_formatter += genFeatMeta.genFeatFormatter(
                    setChn, thisFeatFn, params
                )

        logging.debug(f"{type(featBlock)=}")

        # featBlock = np.hstack([thisFeat(setDat) for thisFeat in featList])
        if featureStore is None:
            featureStore = featBlock
        else:
            featureStore = np.append(featureStore, featBlock, axis=0)

        logging.debug(f"\n{np.shape(featureStore)=} at {iterCnt=}")
        # setDat.apply_function(fn)
        # extract all features from one subject
        # append to a list of all subject
        toc = time.perf_counter()
        logging.debug(f"\nRunning time: {toc - tic:0.4f} seconds")

        if iterCnt % 10 == 0 and iterCnt == 0:
            np.savetxt(
                os.path.join(dirNameOutput, featStoreFN),
                featureStore,
                fmt=feature_formatter,
                delimiter=",",
                header=",".join(feature_names),
            )
            featureStore = None
        elif iterCnt % 10 == 0:
            with open(os.path.join(dirNameOutput, featStoreFN), "a") as f:
                np.savetxt(f, featureStore, fmt=feature_formatter, delimiter=",")
            featureStore = None
    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        assert isinstance(exc_tb, TracebackType)
        fnameExc = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.warning(
            f"{exc_type}\n{fnameExc}\n{exc_tb.tb_lineno}\nSbj Num: "
            + sbjList[iterCnt]
            + "\n"
            + str(e)
        )

if os.path.isfile(os.path.join(dirNameOutput, featStoreFN)):
    with open(os.path.join(dirNameOutput, featStoreFN), "a") as f:
        np.savetxt(f, featureStore, fmt=feature_formatter, delimiter=",")
else:
    np.savetxt(
        os.path.join(dirNameOutput, featStoreFN),
        featureStore,
        fmt=feature_formatter,
        delimiter=",",
        header=",".join(feature_names),
    )

print("Done!")
