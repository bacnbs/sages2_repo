"""
Created on Jun 14, 2023

For both SAGES and Duke dataset

For Duke dataset, only INT files were recorded.

@author: mning
"""
import logging
from typing import Dict
from typing import List
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger("root")


# Entire SAGES Cohort
# support original and modified MoCA
# cannot support eeg subset selection
def getMoCACohort(op_crr) -> pl.DataFrame:
    moca_fn = (
        r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data"
        r"\Baseline_rsEEG_preprocessing\MoCA\vdmoca_subject_interviews.csv"
    )
    moca_df = pl.read_csv(moca_fn)
    if not op_crr:
        moca_df = moca_df.with_columns(
            (
                pl.col("vdmoca_visuospatial_executive")
                + pl.col("vdmoca_naming")
                + pl.col("vdmoca_attention")
                + pl.col("vdmoca_language")
                + pl.col("vdmoca_memory_alternative_items")
                + pl.col("vdmoca_orientation")
            ).alias("MoCA")
        )
    else:
        moca_df = moca_df.with_columns(pl.col("vdmoca").alias("MoCA"))
    moca_df = moca_df.filter(pl.col("timefr") == 0)
    moca_df = moca_df.select(pl.col("*").drop_nans())
    moca_df = moca_df.drop_nulls("MoCA")
    moca_df = moca_df.filter(~(pl.col("MoCA") == 0))
    return moca_df


# Only SAGES EEG ML Project
def getMoCA(op_eo, op_crr) -> Dict[str, Optional[float]]:
    # To add dict as column to pd.Dataframe:
    # df.loc[list(dic), 'a'] = pd.Series(dic)
    moca_scrs: Dict[str, Optional[float]]
    if op_eo and not op_crr:
        # feature store has 85 rows
        moca_scrs = {
            "001": 16,
            "002": 16,
            "003": 7,
            "004": 19,
            "005": 23,
            "006": 27,
            "007": 27,
            "008": 15,
            "009": 22,
            "010": 26,
            "011": 21,
            "012": 26,
            "013": 15,
            "014": 23,
            "015": 21,
            "016": 29,
            "018": 24,
            "019": 24,
            "020": 27,
            "021": 26,
            "022": 28,
            "023": 27,
            "025": 27,
            "027": 22,
            "028": 15,
            "029": 21,
            "030": 25,
            "031": 26,
            "032": 26,
            "033": 29,
            "034": 17,
            "035": 27,
            "036": 28,
            "037": 29,
            "038": 30,
            "039": 24,
            "040": 26,
            "041": 28,
            "043": 24,
            "044": 26,
            "045": 27,
            "047": 25,
            "048": 29,
            "049": 28,
            "050": 22,
            "051": 26,
            "052": 22,
            "053": 30,
            "054": 23,
            "055": 21,
            "056": 15,
            "057": 24,
            "059": 29,
            "060": 12,
            "061": 30,
            "062": 24,
            "063": 28,
            "064": 20,
            "065": 27,
            "066": 24,
            "067": 29,
            "068": 27,
            "069": 24,
            "070": 27,
            "071": 30,
            "072": 24,
            "073": 15,
            "074": 29,
            "075": 29,
            "076": 24,
            "077": 24,
            "078": 26,
            "079": 26,
            "080": 29,
            "082": 21,
            "083": 26,
            "084": 25,
            "085": 27,
            "086": 27,
            "087": 27,
            "088": 25,
            "089": 21,
            "090": 27,
            "091": 25,
            "092": 20,
        }
    elif op_eo and op_crr:
        moca_scrs = {
            "001": 25.6,
            "002": 18.9,
            "003": 14.1,
            "004": 26.4,
            "005": 28.9,
            "006": 26.4,
            "007": 28.9,
            "008": 15.5,
            "009": 26.4,
            "010": 28.9,
            "011": 26.7,
            "012": 27.4,
            "013": 17.6,
            "014": 25.6,
            "015": 26.4,
            "016": 26.4,
            "018": 28.9,
            "019": 28.9,
            "020": 28.9,
            "021": 25.6,
            "022": 25.6,
            "023": 28.9,
            "025": 28.9,
            "027": 25.6,
            "028": 25.6,
            "029": 26.4,
            "030": 27.4,
            "031": 25.6,
            "032": 27.4,
            "033": 26.4,
            "034": 17.6,
            "035": 25.6,
            "036": 27.4,
            "037": 26.4,
            "038": 27.4,
            "039": 27.4,
            "040": 21.2,
            "041": 24.2,
            "043": 30,
            "044": 27.4,
            "045": 26.4,
            "047": 27.4,
            "048": 30,
            "049": 28.9,
            "050": 26.4,
            "051": 28.0,
            "052": 24.2,
            "053": 25.6,
            "054": 27.4,
            "055": 28.9,
            "056": 16.3,
            "057": 25.6,
            "059": 27.4,
            "060": 22.6,
            "061": 30,
            "062": 21.2,
            "063": 28.9,
            "064": 25.6,
            "065": 27.4,
            "066": 24.2,
            "067": 30,
            "068": 26.4,
            "069": 30,
            "070": 26.4,
            "071": 28.9,
            "072": 25.6,
            "073": 22.6,
            "074": 25.6,
            "075": 28.9,
            "076": 30,
            "077": 27.4,
            "078": 26.4,
            "079": 27.4,
            "080": 27.4,
            "082": 17.6,
            "083": 28.9,
            "084": 25.6,
            "085": 27.4,
            "086": 28.9,
            "087": 30,
            "088": 22.6,
            "089": 27.4,
            "090": 28.9,
            "091": 25.6,
            "092": 21.2,
        }
    elif not op_eo and not op_crr:
        # feature store has 88 rows
        # add 38, 39
        moca_scrs = {
            "001": 16,
            "002": 16,
            "003": 7,
            "004": 19,
            "005": 23,
            "006": 27,
            "007": 27,
            "008": 15,
            "009": 22,
            "010": 26,
            "011": 21,
            "012": 26,
            "013": 15,
            "014": 23,
            "015": 21,
            "016": 29,
            "018": 24,
            "019": 24,
            "020": 27,
            "021": 26,
            "022": 28,
            "023": 27,
            "025": 27,
            "027": 22,
            "028": 15,
            "029": 21,
            "030": 25,
            "031": 26,
            "032": 26,
            "033": 29,
            "034": 17,
            "035": 27,
            "036": 28,
            "037": 29,
            "038": 30,
            "039": 24,
            "040": 26,
            "041": 28,
            "043": 24,
            "044": 26,
            "045": 27,
            "047": 25,
            "048": 29,
            "049": 28,
            "050": 22,
            "051": 26,
            "052": 22,
            "053": 30,
            "054": 23,
            "055": 21,
            "056": 15,
            "057": 24,
            "059": 29,
            "060": 12,
            "061": 30,
            "062": 24,
            "063": 28,
            "064": 20,
            "065": 27,
            "066": 24,
            "067": 29,
            "068": 27,
            "069": 24,
            "070": 27,
            "071": 30,
            "072": 24,
            "073": 15,
            "074": 29,
            "075": 29,
            "076": 24,
            "077": 24,
            "078": 26,
            "079": 26,
            "080": 29,
            "082": 21,
            "083": 26,
            "084": 25,
            "085": 27,
            "086": 27,
            "087": 27,
            "088": 25,
            "089": 21,
            "090": 27,
            "091": 25,
            "092": 20,
            "017": 20,
            "026": 26,
            "042": 26,
        }
    elif not op_eo and op_crr:
        moca_scrs = {
            "001": 25.6,
            "002": 18.9,
            "003": 14.1,
            "004": 26.4,
            "005": 28.9,
            "006": 26.4,
            "007": 28.9,
            "008": 15.5,
            "009": 26.4,
            "010": 28.9,
            "011": 26.7,
            "012": 27.4,
            "013": 17.6,
            "014": 25.6,
            "015": 26.4,
            "016": 26.4,
            "018": 28.9,
            "019": 28.9,
            "020": 28.9,
            "021": 25.6,
            "022": 25.6,
            "023": 28.9,
            "025": 28.9,
            "027": 25.6,
            "028": 25.6,
            "029": 26.4,
            "030": 27.4,
            "031": 25.6,
            "032": 27.4,
            "033": 26.4,
            "034": 17.6,
            "035": 25.6,
            "036": 27.4,
            "037": 26.4,
            "038": 27.4,
            "039": 27.4,
            "040": 21.2,
            "041": 24.2,
            "043": 30,
            "044": 27.4,
            "045": 26.4,
            "047": 27.4,
            "048": 30,
            "049": 28.9,
            "050": 26.4,
            "051": 28.0,
            "052": 24.2,
            "053": 25.6,
            "054": 27.4,
            "055": 28.9,
            "056": 16.3,
            "057": 25.6,
            "059": 27.4,
            "060": 22.6,
            "061": 30,
            "062": 21.2,
            "063": 28.9,
            "064": 25.6,
            "065": 27.4,
            "066": 24.2,
            "067": 30,
            "068": 26.4,
            "069": 30,
            "070": 26.4,
            "071": 28.9,
            "072": 25.6,
            "073": 22.6,
            "074": 25.6,
            "075": 28.9,
            "076": 30,
            "077": 27.4,
            "078": 26.4,
            "079": 27.4,
            "080": 27.4,
            "082": 17.6,
            "083": 28.9,
            "084": 25.6,
            "085": 27.4,
            "086": 28.9,
            "087": 30,
            "088": 22.6,
            "089": 27.4,
            "090": 28.9,
            "091": 25.6,
            "092": 21.2,
            "017": 26.4,
            "026": 28.9,
            "042": 25.6,
        }
    return moca_scrs


def getSpatialEx() -> Dict[str, Optional[float]]:
    # only EO subjects
    # To add dict as column to pd.Dataframe:
    # df.loc[list(dic), 'a'] = pd.Series(dic)
    # feature store has 85 rows
    spatialEx_scrs: Dict[str, Optional[float]] = {
        "001": 4,
        "002": 3,
        "003": 1,
        "004": 4,
        "005": 4,
        "006": 4,
        "007": 4,
        "008": 4,
        "009": 3,
        "010": 5,
        "011": 0,
        "012": 4,
        "013": 3,
        "014": 5,
        "015": 4,
        "016": 4,
        "018": 4,
        "019": 4,
        "020": 4,
        "021": 4,
        "022": 4,
        "023": 4,
        "025": 5,
        "027": 4,
        "028": 4,
        "029": 3,
        "030": 4,
        "031": 4,
        "032": 5,
        "033": 4,
        "034": 3,
        "035": 5,
        "036": 4,
        "037": 3,
        "038": 5,
        "039": 5,
        "040": 3,
        "041": 4,
        "043": 5,
        "044": 5,
        "045": 3,
        "047": 4,
        "048": 5,
        "049": 5,
        "050": 5,
        "051": 0,
        "052": 4,
        "053": 3,
        "054": 4,
        "055": 4,
        "056": 3,
        "057": 3,
        "059": 5,
        "060": 4,
        "061": 5,
        "062": 4,
        "063": 5,
        "064": 3,
        "065": 3,
        "066": 3,
        "067": 5,
        "068": 3,
        "069": 5,
        "070": 3,
        "071": 4,
        "072": 4,
        "073": 3,
        "074": 4,
        "075": 5,
        "076": 5,
        "077": 4,
        "078": 3,
        "079": 4,
        "080": 4,
        "082": 3,
        "083": 4,
        "084": 3,
        "085": 4,
        "086": 5,
        "087": 5,
        "088": 2,
        "089": 4,
        "090": 4,
        "091": 3,
        "092": 2,
        "017": 4,
        "026": 5,
        "042": 4,
    }
    return spatialEx_scrs


def getAttention() -> Dict[str, Optional[float]]:
    # only EO subjects
    # To add dict as column to pd.Dataframe:
    # df.loc[list(dic), 'a'] = pd.Series(dic)
    # feature store has 85 rows
    atten_scrs: Dict[str, Optional[float]] = {
        "001": 6,
        "002": 4,
        "003": 2,
        "004": 4,
        "005": 6,
        "006": 5,
        "007": 6,
        "008": 0,
        "009": 6,
        "010": 6,
        "011": 6,
        "012": 6,
        "013": 2,
        "014": 6,
        "015": 5,
        "016": 6,
        "018": 6,
        "019": 6,
        "020": 6,
        "021": 5,
        "022": 6,
        "023": 6,
        "025": 6,
        "027": 5,
        "028": 5,
        "029": 6,
        "030": 6,
        "031": 6,
        "032": 5,
        "033": 6,
        "034": 3,
        "035": 5,
        "036": 6,
        "037": 6,
        "038": 5,
        "039": 5,
        "040": 5,
        "041": 6,
        "043": 6,
        "044": 6,
        "045": 6,
        "047": 6,
        "048": 6,
        "049": 6,
        "050": 6,
        "051": 6,
        "052": 5,
        "053": 6,
        "054": 6,
        "055": 6,
        "056": 1,
        "057": 6,
        "059": 6,
        "060": 6,
        "061": 6,
        "062": 3,
        "063": 6,
        "064": 6,
        "065": 6,
        "066": 6,
        "067": 6,
        "068": 6,
        "069": 6,
        "070": 6,
        "071": 6,
        "072": 5,
        "073": 6,
        "074": 6,
        "075": 6,
        "076": 6,
        "077": 6,
        "078": 6,
        "079": 6,
        "080": 5,
        "082": 3,
        "083": 6,
        "084": 6,
        "085": 6,
        "086": 6,
        "087": 6,
        "088": 6,
        "089": 6,
        "090": 6,
        "091": 6,
        "092": 6,
        "017": 6,
        "026": 6,
        "042": 6,
    }
    return atten_scrs


def getMemory() -> Dict[str, Optional[float]]:
    # only EO subjects
    # To add dict as column to pd.Dataframe:
    # df.loc[list(dic), 'a'] = pd.Series(dic)
    # feature store has 85 rows
    mem_scrs: Dict[str, Optional[float]] = {
        "001": 3,
        "002": 0,
        "003": 0,
        "004": 0,
        "005": 3,
        "006": 2,
        "007": 3,
        "008": 0,
        "009": 2,
        "010": 1,
        "011": 4,
        "012": 5,
        "013": 0,
        "014": 0,
        "015": 1,
        "016": 5,
        "018": 0,
        "019": 1,
        "020": 0,
        "021": 3,
        "022": 0,
        "023": 2,
        "025": 2,
        "027": 3,
        "028": 0,
        "029": 0,
        "030": 4,
        "031": 4,
        "032": 5,
        "033": 2,
        "034": 0,
        "035": 3,
        "036": 0,
        "037": 2,
        "038": 5,
        "039": 0,
        "040": 2,
        "041": 1,
        "043": 0,
        "044": 3,
        "045": 3,
        "047": 4,
        "048": 5,
        "049": 0,
        "050": 3,
        "051": 3,
        "052": 3,
        "053": 4,
        "054": 0,
        "055": 3,
        "056": 0,
        "057": 0,
        "059": 5,
        "060": 1,
        "061": 5,
        "062": 5,
        "063": 3,
        "064": 1,
        "065": 1,
        "066": 3,
        "067": 4,
        "068": 4,
        "069": 2,
        "070": 5,
        "071": 3,
        "072": 4,
        "073": 0,
        "074": 5,
        "075": 5,
        "076": 0,
        "077": 1,
        "078": 2,
        "079": 2,
        "080": 2,
        "082": 2,
        "083": 3,
        "084": 3,
        "085": 5,
        "086": 4,
        "087": 0,
        "088": 1,
        "089": 2,
        "090": 2,
        "091": 3,
        "092": 3,
        "017": 2,
        "026": 5,
        "042": 4,
    }
    return mem_scrs


def getMMSE() -> Dict[str, Optional[float]]:
    # To add dict as column to pd.Dataframe:
    # df.loc[list(dic), 'a'] = pd.Series(dic)
    mmse_scrs: Dict[str, Optional[float]] = {
        "0115": 29,
        "0182": 27,
        "0190": None,
        "0192": 26,
        "023": 29,
        "024": 29,
        "025": 19,
        "027": 26,
        "028": 26,
        "032": 26,
        "033": 28,
        "034": 30,
        "038": 30,
        "040": 24,
        "043": 29,
        "044": 29,
        "045": 30,
        "046": 30,
        "047": 28,
        "049": 28,
        "050": 27,
        "051": 28,
        "056": 28,
        "057": 29,
        "059": 29,
        "061": 29,
        "062": 25,
        "064": 27,
        "066": 23,
        "067": 29,
        "068": 29,
        "069": 28,
        "074": 25,
        "077": 25,
        "079": 29,
        "083": 21,
        "085": 29,
        "092": 26,
        "093": 30,
        "094": 30,
        "097": 29,
        "098": 29,
        "102": 23,
        "110": 28,
        "112": 28,
        "113": 30,
        "185": 30,
        "193": 25,
        "194": 28,
        "195": 28,
        "198": 29,
        "199": 30,
        "201": 29,
    }
    return mmse_scrs


def conversion_MMSE_MoCA(mmse_score) -> Optional[float]:
    if mmse_score == 30:
        return 28.5
    elif mmse_score == 29:
        return 25.5
    elif mmse_score == 28:
        return 24
    elif mmse_score == 27:
        return 22.5
    elif mmse_score == 26:
        return 21
    elif mmse_score == 25:
        return 20
    elif mmse_score == 24:
        return 18.5
    elif mmse_score == 23:
        return 17
    elif mmse_score == 22:
        return 16
    elif mmse_score == 21:
        return 14.5
    elif mmse_score == 20:
        return 13
    elif mmse_score == 19:
        return 12
    elif mmse_score == 18:
        return 11.5
    elif mmse_score == 17:
        return 9
    elif mmse_score == 16:
        return 8
    elif mmse_score == 15:
        return 7
    elif mmse_score == 14:
        return 6
    elif mmse_score == 12:
        return 5
    elif mmse_score == 11:
        return 4
    elif mmse_score == 8:
        return 3
    elif mmse_score == 5:
        return 2
    elif mmse_score == 2:
        return 1
    elif mmse_score == 0:
        return 0
    return None


def conversion_MoCA_MMSE(moca_score) -> Optional[float]:
    if moca_score == 30:
        return 30
    elif moca_score == 29:
        return 30
    elif moca_score == 28:
        return 30
    elif moca_score == 27:
        return 30
    elif moca_score == 26:
        return 29
    elif moca_score == 25:
        return 29
    elif moca_score == 24:
        return 28
    elif moca_score == 23:
        return 27
    elif moca_score == 22:
        return 27
    elif moca_score == 21:
        return 26
    elif moca_score == 20:
        return 25
    elif moca_score == 19:
        return 24
    elif moca_score == 18:
        return 24
    elif moca_score == 17:
        return 23
    elif moca_score == 16:
        return 22
    elif moca_score == 15:
        return 21
    elif moca_score == 14:
        return 21
    elif moca_score == 13:
        return 20
    elif moca_score == 12:
        return 19
    elif moca_score == 11:
        return 18
    elif moca_score == 10:
        return 18
    elif moca_score == 9:
        return 17
    elif moca_score == 8:
        return 16
    elif moca_score == 7:
        return 15
    elif moca_score == 6:
        return 14
    elif moca_score == 5:
        return 12
    elif moca_score == 4:
        return 11
    elif moca_score == 3:
        return 8
    elif moca_score == 2:
        return 5
    elif moca_score == 1:
        return 2
    elif moca_score == 0:
        return 0
    return None


def convert_MMSE_MoCA(mmse: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    moca_scrs: Dict[str, Optional[float]] = {
        k: conversion_MMSE_MoCA(v) for (k, v) in mmse.items()
    }

    return moca_scrs


def convert_MMSE_MoCA_List(mmse: List[Optional[float]]) -> List[Optional[float]]:
    moca_scrs: List[Optional[float]] = [conversion_MMSE_MoCA(v) for v in mmse]
    return moca_scrs


def convert_MoCA_MMSE(moca: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
    mmse_scrs: Dict[str, Optional[float]] = {
        k: conversion_MoCA_MMSE(v) for (k, v) in moca.items()
    }

    return mmse_scrs
