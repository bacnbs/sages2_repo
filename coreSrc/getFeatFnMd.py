"""
Created on May 10, 2023

@author: mning

Note: all must accept same input arguments and yield same output types
"""
import logging
from typing import Callable

from EEGFeaturesExtraction.coreSrc.features import getInfoMetrics
from EEGFeaturesExtraction.coreSrc.features import getLyapunovExpMd
from EEGFeaturesExtraction.coreSrc.features import getSignalLineLenMd
from EEGFeaturesExtraction.coreSrc.features import getSpecStatsMd
from EEGFeaturesExtraction.coreSrc.features import getStatsMd


# import getMin
# import getSTD
logger = logging.getLogger("root")

typeFnList = list[Callable]


def getFeatFn() -> typeFnList:
    featFnList = []
    featFnList.append(getSignalLineLenMd.getSignalLineLen)
    featFnList.append(getStatsMd.getKurtosis)
    # featFnList.append(getInfoMetrics.getSampleEntropy)
    featFnList.append(getSpecStatsMd.getMTSpectrogram)
    # featFnList.append(LyE_R)
    # featFnList.append(getMaxMd.getMax)
    # featFnList.append(getMinMd.getMin)
    # featFnList.append(getSTDMd.getSTD)

    return featFnList


#   featFnList.append(getMin)
#   featFnList.append(getSTD)


def getFeatFnSbj() -> typeFnList:
    featFnList = []
    # featFnList.append(getSignalLineLenMd.getSignalLineLen)
    # featFnList.append(getStatsMd.getKurtosis)
    # featFnList.append(getInfoMetrics.getSampleEntropy)
    featFnList.append(getSpecStatsMd.getMTSpectrogramSbj)
    # featFnList.append(LyE_R)
    # featFnList.append(getMaxMd.getMax)
    # featFnList.append(getMinMd.getMin)
    # featFnList.append(getSTDMd.getSTD)

    return featFnList


def getPSDFnSbj() -> typeFnList:
    featFnList = []
    # featFnList.append(getSignalLineLenMd.getSignalLineLen)
    # featFnList.append(getStatsMd.getKurtosis)
    # featFnList.append(getInfoMetrics.getSampleEntropy)
    featFnList.append(getSpecStatsMd.computePSD)
    # featFnList.append(LyE_R)
    # featFnList.append(getMaxMd.getMax)
    # featFnList.append(getMinMd.getMin)
    # featFnList.append(getSTDMd.getSTD)

    return featFnList


def getBPFnSbj() -> typeFnList:
    featFnList = []
    featFnList.append(getSpecStatsMd.computeBandpower)
    return featFnList


def getBPSTDFnSbj() -> typeFnList:
    featFnList = []
    featFnList.append(getSpecStatsMd.computeBandpowerSTD)
    return featFnList


def getSampleEntropyFn() -> typeFnList:
    featFnList = []
    featFnList.append(getInfoMetrics.getSampleEntropy)
    # featFnList.append(getSpecStatsMd.getMTSpectrogram)
    # featFnList.append(LyE_R)
    # featFnList.append(getMaxMd.getMax)
    # featFnList.append(getMinMd.getMin)
    # featFnList.append(getSTDMd.getSTD)

    return featFnList


#   featFnList.append(getMin)
#   featFnList.append(getSTD)


def getPermutationEntropyFn() -> typeFnList:
    featFnList = []
    featFnList.append(getInfoMetrics.getPermutationEntropy)
    return featFnList


def getLyEFn() -> typeFnList:
    featFnList = []
    featFnList.append(getLyapunovExpMd.getLyapunovExp)
    return featFnList
