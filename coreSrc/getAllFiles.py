"""
Created on May 10, 2023

@author: mning
"""
import glob
import logging
import os
import re

logger = logging.getLogger("root")


def getAllFilesEndingRegex(dirName: str, matchStr: str):
    """Find all files that end with given substring within given directory
    E.g.:
        dirName = ('R:\Studies_Investigator Folders\Shafi'
            '\2018P-000227_SAGES I and II\Data'
            '\Baseline_rsEEG_preprocessing\DATA_preprocessed_MN_Copy')
        matchStr = '_EC_S06_59.set'

    """

    os.chdir(dirName)

    regPattern = os.path.join("**", "*" + matchStr)

    res = glob.glob(regPattern, recursive=False)

    sbjList = []
    fileList = []
    fileCnt = 0

    for fName in res:
        logger.debug(f"File Name {fName}")
        thisSbStr = fName.split(os.sep)[0].split("_")[1]
        # logger.debug(thisSbStr)
        reSrchRslt = re.search(r"\d+", thisSbStr)
        assert reSrchRslt is not None
        sbjList.append(reSrchRslt.group())
        fileList.append(fName)
        fileCnt += 1

    logger.debug(f"Total number of files: {fileCnt}")

    return fileList, sbjList


def getAllFilesBeginEndingRegex(dirName: str, startStr: str, endStr: str):
    """Find all files that start and end with given substring within given directory
    E.g.:
        dirName = ('R:\Studies_Investigator Folders\Shafi'
        '\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing'
        '\DATA_preprocessed_MN_Copy')
        matchStr = '_EC_S06_59.set'

    """

    sbjList = []
    fileList = []
    fileCnt = 0

    os.chdir(dirName)

    regPattern = os.path.join("**", startStr + "*" + endStr)

    for fName in glob.glob(regPattern):
        logger.debug(f"File Name {fName}")
        thisSbStr = fName.split(os.sep)[0].split("_")[1]
        # logger.debug(thisSbStr)
        reSrchRslt = re.search(r"\d+", thisSbStr)
        assert reSrchRslt is not None
        sbjList.append(reSrchRslt.group())
        fileList.append(fName)
        fileCnt += 1

    logger.debug(f"Total number of files: {fileCnt}")

    return fileList, sbjList


def getAllFilesEndingRegex_AD(dirName: str, matchStr: str):
    """Find all files that end with given substring within given directory
    E.g.:
        dirName = ('R:\Studies_Investigator Folders\Shafi'
            '\2018P-000227_SAGES I and II\Data'
            '\Baseline_rsEEG_preprocessing\DATA_preprocessed_MN_Copy')
        matchStr = '_EC_S06_59.set'

    """

    os.chdir(dirName)

    regPattern = os.path.join("**", "*" + matchStr)

    res = glob.glob(regPattern, recursive=True)

    sbjList = []
    fileList = []
    fileCnt = 0

    for fName in res:
        print(fName)
        logger.debug(f"File Name {fName}")
        thisSbStr = fName.split(os.sep)[1]
        sbjList.append(thisSbStr)
        fileList.append(os.path.join(dirName, fName))
        fileCnt += 1

    logger.debug(f"Total number of files: {fileCnt}")

    return fileList, sbjList
