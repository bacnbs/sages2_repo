"""
STATUS:
  WIP.

FOR PUBLICATION:
  Unknown.

SYNTAX:
  Script.

DESCRIPTION:
  Performs classification of powers of different frequency bands and plot
  different performance metrics. Additional debugging information are also
  saved. Define training/test split for SAGES and Duke Data Sets.  Used for
  model validation.

PIPELINE SUMMARY:


RESTRICTION:
  1) Feature selection option is not properly supported for bootstrapping or
    over-sampling. Feature selection should be done before over-sampling.
  2) Other combinations of options may not be working properly or incorrectly
    implemented.
  3) Figure of NSC feature selection is not supported for `op_Training == "BS"`.
  4) `op_ROI` not supported.

INPUT FILES:
  `featFN_EO_SAGES` - file name of feature store for SAGES Data Set-
    Eyes Open condition.
  `featFN_EC_SAGES` - file name of feature store for SAGES Data Set-
    Eyes Closed condition.
  `featFN_EO_DUKE` - file name of feature store for Duke Data Set-
    Eyes Open condition.
  `featFN_EC_DUKE` - file name of feature store for Duke Data Set-
    Eyes Closed condition.

OPTIONS:
  `op_Training` = {'DUKE','SAGES','Combined','BS'}.
    -'BS': Bootstrap SAGES Data Set as the training set.
    -`BSTst`: Bootstrap DUKE Data Set as the test set.
    -`BSErr`: Report full-sample results with CIs derived from BS-ing
      test set.
    -`SAGES`: SAGES is the training set and Duke is the test set.
    -`DUKE`: Duke is the training set and SAGES is the test set.
    -`CV`': Combine SAGES and Duke Data Sets and cross-validate it.
  `op_Scl_CogAssmnt` = {0, 1, 2, 3}.
    Only applies to cognitive assessments, not EEG features.
    -0 : z-score using DUKE sample
    -1 : none
    -2 : Distance to the log10 of the median of cognitive assessments.
    -3 : z-score using SWEDEN Population
    -4 : z-score using SAGES sample
  `dist_Medn` = {0, 1, 2, 3, 4, 5}.
    Only applies to EEG features, not cognitive assessments. Median
    refers to the median of the whole data set.
    -0 : Z-transform.
    -1 : Distance to the median.
    -2 : Distance to the log10 of the median of each column.
    -3 : Distance to the log10 of the median of all columns.
    -4 : Distance to the median of arc sinh transformed data.
    -5 : Distance to the 10*log10 of the median of each column.
  `op_CogAssmnt` = {0, 1, 2, 3, 4}.
    -0 : None.
    -1 : MoCA.
    -2 : GCP.
    -3 : MoCA Sub-domain scores. To be further specified by op_sub.
    -4 : MMSE
  `op_crr` = {0 ,1}
    Original vs corrected MoCA for SAGES.
    -0 : Original (sum)
    -1 : Corrected MoCA (pro-rated)
  `op_sub` = {1, 2, 3, 4}.
    -0 : Visuospatial
    -1 : Attention
    -2 : Memory
    -3 : Sum of 3 subdomains
  `op_Oversampling` = {0, 1, 2}.
    -0 : No oversampling
    -1 : Random oversampling with smoothing
    -2 : SMOTE
  `op_FeatSelect` = {0, 1, 2, 3, 4}
    -0 : None
    -1 : Select K Best
    -2 : Mutual Information
    -3 : Ridge Classifier, forward feature selection
    -4 : NSC Manhattan, forward feature selection
  `num_F` = positive integer.
    For op_FeatSelect. Specifies the number of features to be selected.
  `op_10_20` = {0, 1}
    -0 : Use all 64-channel
    -1 : Limit to 32 channel for compatibility with Duke Data Set
  `op_ROI` = {0, 1}
    -0 : Use features extract from individual channels
    -1 : Use features averaged over channels within ROI
  `op_Sex` = {0, 1}
    -0 : Don't use sex of subjects as feature
    -1 : Use sex of subjects as feature
  `op_Age` = {0, 1}
    -0 : Don't use age of subjects as feature
    -1 : Use age of subjects as feature

RETURNED VARIABLES:
  None.

FIGURES SAVED:
  1) `bs_lda_cv_fn` - histograms of bootstrapped performance metrics
    for LDA CV model.
      Only for `op_Training` = "BS".
  2) `bs_lda_lw_fn` - histograms of bootstrapped performance metrics
    for LDA LW model.
      Only for `op_Training` = "BS".
  3) `bs_nsc_mh_fn` - histograms of bootstrapped performance metrics
    for NSC Manhattan model.
      Only for `op_Training` = "BS".
  4) `perf_mean_fig` - bar plots of different performance metrics of
    different models, using mean of bootstrapped samples.

TEXT FILES SAVED:
  `feat_sel_SAGES_FN` - list of features selected during feature selection.
  `coefFN` - csv of coefficients for 1) LDA OA, 2) LDA LW and 3) Logistic
    Regression with L2 regularization.
  -`perf_csv_fn` - csv of performance metrics.

REPRODUCIBILITY:
  Instruction for reproducing figures.

SEE ALSO:
  Compare and contrast similar scripts here.

AUTHOR: Matthew Ning, Ph.D., Beth Israel Deaconess Medical Center.

LOG:
  12/20/2023: Add documentation.

TODO:
  1) Add pytests and data validation (pydantic).
  2) Wrap some blocks of codes in new functions.
  3) Simplify nested if-else blocks.
  4) Add comments to make code review easier.
  5) Save performance metrics to csv files. DONE.
  6) Plot/save training performance metrics.
  7) Make font size bigger in figures. DONE.
  8) Remove support for `op_LowMoCA`.

@author: mning
"""
import os
from _io import TextIOWrapper
from contextlib import nullcontext
from typing import Any
from typing import ContextManager
from typing import Dict
from typing import TextIO
from typing import Union
from unittest import TestCase

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from EEGFeaturesExtraction.coreSrc.chc_lvl_simulation_Md import get_Theor_Chc
from EEGFeaturesExtraction.coreSrc.estimators import _NSC
from EEGFeaturesExtraction.coreSrc.features import get1020Md
from EEGFeaturesExtraction.coreSrc.features import getAgeMd
from EEGFeaturesExtraction.coreSrc.features import getDeliriumStatusMd
from EEGFeaturesExtraction.coreSrc.features import getGCPMd
from EEGFeaturesExtraction.coreSrc.features import getMoCAMD
from EEGFeaturesExtraction.coreSrc.features import getSexMd
from EEGFeaturesExtraction.coreSrc.getModelFnMd import getEnsembleModelFn
from EEGFeaturesExtraction.coreSrc.getModelFnMd import getModelCVFn
from EEGFeaturesExtraction.utils import get_Clrs_Md
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from scipy.stats import t
from scipy.stats import zscore
from sklearn import metrics
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model._ridge import RidgeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

plt.set_loglevel(level="warning")

## Option variables here
op_Local = 0
testCase = 0
op_Training = "BSErr"
num_rep_bs = 2000
dist_Medn = 2
op_Oversampling = 0
ch_name = "EOEC"
op_Scl_CogAssmnt = 4
op_CogAssmnt = 0
op_sub = 0
op_ManualFeat = 1
op_Ensemble = 0
op_Reref = "car"
op_LowMoCA = 0
op_crr = 1
op_FeatSelect = 0
op_10_20 = 1
op_mV = 1
op_ROI = 0
num_F = 10
op_Sex = 0
op_Age = 0
op_Baseline = "MoCAAlone"
relativeFOp = True

## Option Strings Here
if op_Oversampling == 1:
    ch_name += "_OSRandom"
elif op_Oversampling == 2:
    ch_name += "_OSSMOTE"

if op_Sex:
    ch_name = ch_name + "_Sex"

if op_Age:
    ch_name = ch_name + "_Age"

if op_CogAssmnt == 1:
    if op_crr:
        moca_str = "MoCACorrected"
    else:
        moca_str = "MoCA"
elif op_CogAssmnt == 2:
    moca_str = "GCP"
elif op_CogAssmnt == 3:
    if op_sub == 1:
        moca_str = "VisuoSpa"
    elif op_sub == 2:
        moca_str = "Attention"
    elif op_sub == 3:
        moca_str = "Memory"
    elif op_sub == 4:
        moca_str = "Sum3SubMoCA"
elif op_CogAssmnt == 4:
    moca_str = "MMSE"
elif op_CogAssmnt == 0:
    moca_str = "NoMoCA"

if op_Scl_CogAssmnt == 0:
    moca_str += "_ZScore"
elif op_Scl_CogAssmnt == 1:
    moca_str += "_Untrnsfrm"
elif op_Scl_CogAssmnt == 2:
    moca_str += "_DistMdn"
elif op_Scl_CogAssmnt == 3:
    moca_str += "_ZscoresSwedenPop"
elif op_Scl_CogAssmnt == 4:
    moca_str += "_ZscoresSAGESPop"

if op_ManualFeat:
    manual_str = "Manual"
else:
    manual_str = "Automatic"

if op_LowMoCA and op_CogAssmnt:
    moca_str += "Low"
elif not op_LowMoCA and op_CogAssmnt:
    moca_str += "All"
# relativeFOp is for getSpecStatsMd.bandpower

if op_FeatSelect == 1:
    ch_name += "_FTest"
elif op_FeatSelect == 2:
    ch_name += "_MI"
elif op_FeatSelect == 3:
    ch_name += "_Frwrd"
elif op_FeatSelect == 4:
    ch_name += "_NSCFrwrd"
elif op_FeatSelect == 5:
    ch_name += "_L1"

if op_FeatSelect:
    ch_name += f"_{num_F}"

if dist_Medn == 1:
    moca_str += "DistMdn"
elif dist_Medn == 2:
    moca_str += "DistMdnLog10"
elif dist_Medn == 3:
    moca_str += "DistMdnWholeLog10"
elif dist_Medn == 4:
    moca_str += "DictMdnArcSinH"
elif dist_Medn == 5:
    moca_str += "DistMdn10Log10"
else:
    moca_str += "ZTrnsfrm"

if op_10_20 and not op_ROI:
    mnt_str = "mnt1020"
elif op_ROI:
    mnt_str = "ROI"
else:
    mnt_str = "64chn"

if relativeFOp:
    relative_str = "Relative"
else:
    relative_str = "Absolute"

fn_opt = f"{op_Reref}_{op_Training}_{moca_str}" f"_{manual_str}_{ch_name}_{mnt_str}"


## File names and directories here
figDir = (
    r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data"
    "\Baseline_rsEEG_preprocessing\DATA_visualization\Group"
    "\Performances"
)

## For debugging
csvTestDir = (
    r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data"
    "\Baseline_rsEEG_preprocessing\csv_tests"
)

if op_Local:
    featFN_EO_SAGES = (
        rf"C:\Users\mning\Documents\CNBS_Projects"
        rf"\SAGES I and II\Data\Baseline_rsEEG_preprocessing"
        rf"\FeatureStores\featureStore_Sbj_BP_Bands_EO_"
        rf"{relative_str}_car_mV.csv"
    )
    featFN_EC_SAGES = (
        rf"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II"
        rf"\Data\Baseline_rsEEG_preprocessing\FeatureStores"
        rf"\featureStore_Sbj_BP_Bands_EC_{relative_str}_car_mV.csv"
    )
    featFN_EO_DUKE = (
        rf"R:\Studies_Investigator Folders\Shafi"
        rf"\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA"
        rf"\FeatureStores"
        rf"\featureStore_Sbj_BP_Bands_EO_{relative_str}_car_mV.csv"
    )
    featFN_EC_DUKE = (
        rf"R:\Studies_Investigator Folders\Shafi"
        rf"\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA"
        rf"\FeatureStores"
        rf"\featureStore_Sbj_BP_Bands_EC_{relative_str}_car_mV.csv"
    )
else:
    featFN_EO_SAGES = (
        rf"R:\Studies_Investigator Folders\Shafi"
        rf"\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing"
        rf"\FeatureStores"
        rf"\featureStore_Sbj_BP_Bands_EO_{relative_str}_car_mV.csv"
    )
    featFN_EC_SAGES = (
        rf"R:\Studies_Investigator Folders\Shafi"
        rf"\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing"
        rf"\FeatureStores"
        rf"\featureStore_Sbj_BP_Bands_EC_{relative_str}_car_mV.csv"
    )
    featFN_EO_DUKE = (
        rf"R:\Studies_Investigator Folders\Shafi"
        rf"\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA"
        rf"\FeatureStores"
        rf"\featureStore_Sbj_BP_Bands_EO_{relative_str}_car_mV.csv"
    )
    featFN_EC_DUKE = (
        rf"R:\Studies_Investigator Folders\Shafi"
        rf"\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA"
        rf"\FeatureStores"
        rf"\featureStore_Sbj_BP_Bands_EC_{relative_str}_car_mV.csv"
    )

## Read feature store into panda dataframes
## Add pydantic
feats_EO_SAGES_DF = pd.read_csv(
    featFN_EO_SAGES,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)
feats_EC_SAGES_DF = pd.read_csv(
    featFN_EC_SAGES,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)
feats_EO_DUKE_DF = pd.read_csv(
    featFN_EO_DUKE,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)
feats_EC_DUKE_DF = pd.read_csv(
    featFN_EC_DUKE,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)

## Get cognitive assessments
if op_CogAssmnt == 1:
    cogScore_SAGES_dict = getMoCAMD.getMoCA(1, op_crr)
    cogScore_DUKE_dict = getMoCAMD.getMMSE()
    cogScore_DUKE_dict = getMoCAMD.convert_MMSE_MoCA(cogScore_DUKE_dict)
elif op_CogAssmnt == 2:
    cogScore_SAGES_dict = getGCPMd.getGCP(1)
elif op_CogAssmnt == 4:
    temp_dict = getMoCAMD.getMoCA(1, op_crr)
    temp_dict_round = {k: round(v) for (k, v) in temp_dict.items()}
    cogScore_SAGES_dict = getMoCAMD.convert_MoCA_MMSE(temp_dict_round)
    cogScore_DUKE_dict = getMoCAMD.getMMSE()

## Drop `SetFile` column from dataframes.
new_psd_EO_SAGES_DF = feats_EO_SAGES_DF.drop("SetFile", axis="columns", inplace=False)
new_psd_EC_SAGES_DF = feats_EC_SAGES_DF.drop("SetFile", axis="columns", inplace=False)
new_psd_EO_DUKE_DF = feats_EO_DUKE_DF.drop("SetFile", axis="columns", inplace=False)
new_psd_EC_DUKE_DF = feats_EC_DUKE_DF.drop("SetFile", axis="columns", inplace=False)

del feats_EO_SAGES_DF, feats_EC_SAGES_DF, feats_EO_DUKE_DF, feats_EC_DUKE_DF

## Set `# SbjID` and `TrialNum` columns as indices.
new_psd_EO_SAGES_DF.set_index(["# SbjID", "TrialNum"], inplace=True)
new_psd_EC_SAGES_DF.set_index(["# SbjID", "TrialNum"], inplace=True)
new_psd_EO_DUKE_DF.set_index(["# SbjID", "TrialNum"], inplace=True)
new_psd_EC_DUKE_DF.set_index(["# SbjID", "TrialNum"], inplace=True)

# rename columns before performing operations
# numpy.ndarray
# alpha_col_names_orig = psd_df.columns.values
#
# freqs = list(zip(*(col.split('_') for col in new_psd_EO_SAGES_DF)))[-1]
# freqs_flt = [float(f) for f in freqs]
# freqs_flt_arr = np.asarray(freqs_flt)
# cols_freqs = np.where(np.logical_and(freqs_flt_arr>=freq_min,
# freqs_flt_arr<=freq_max))
# new_psd_EO_SAGES_DF = new_psd_EO_SAGES_DF.iloc[:,cols_freqs[0]]
#
# freqs = list(zip(*(col.split('_') for col in new_psd_EC_SAGES_DF)))[-1]
# freqs_flt = [float(f) for f in freqs]
# freqs_flt_arr = np.asarray(freqs_flt)
# cols_freqs = np.where(np.logical_and(freqs_flt_arr>=freq_min,
# freqs_flt_arr<=freq_max))
# new_psd_EC_SAGES_DF = new_psd_EC_SAGES_DF.iloc[:,cols_freqs[0]]
#
# freqs = list(zip(*(col.split('_') for col in new_psd_EO_DUKE_DF)))[-1]
# freqs_flt = [float(f) for f in freqs]
# freqs_flt_arr = np.asarray(freqs_flt)
# cols_freqs = np.where(np.logical_and(freqs_flt_arr>=freq_min,
# freqs_flt_arr<=freq_max))
# new_psd_EO_DUKE_DF = new_psd_EO_DUKE_DF.iloc[:,cols_freqs[0]]
#
# freqs = list(zip(*(col.split('_') for col in new_psd_EC_DUKE_DF)))[-1]
# freqs_flt = [float(f) for f in freqs]
# freqs_flt_arr = np.asarray(freqs_flt)
# cols_freqs = np.where(np.logical_and(freqs_flt_arr>=freq_min,
# freqs_flt_arr<=freq_max))
# new_psd_EC_DUKE_DF = new_psd_EC_DUKE_DF.iloc[:,cols_freqs[0]]

## Get sexes of subjects.
sex_dict_SAGES = getSexMd.getSex(1)
sex_dict_DUKE = getSexMd.getSexDuke()

## Get ages of subjects.
age_dict_SAGES = getAgeMd.getAge(1)
age_dict_DUKE = getAgeMd.getAgeDuke()

## Encode sexes as 0 or 1
for key, value in sex_dict_SAGES.items():
    sex_dict_SAGES[key] = 1 if value == "M" else 0

for key, value in sex_dict_DUKE.items():
    sex_dict_DUKE[key] = 1 if value == "M" else 0

# n_freq = len(freqs)
#
# mean_alpha_copy_df = psd_df.copy()
#
# mean_alpha_copy_df.rename(
#     columns=dict(
#         zip(
#             alpha_col_names_orig.tolist(),
#             chn_names)
#         ),
#     inplace=True)

## Merge EO and EC dataframes into one and rename columns
## Add pydantic and data validation here.
new_psd_SAGES_DF = new_psd_EO_SAGES_DF.join(
    new_psd_EC_SAGES_DF,
    on=["# SbjID", "TrialNum"],
    how="inner",
    lsuffix="_EO",
    rsuffix="_EC",
)
new_psd_DUKE_DF = new_psd_EO_DUKE_DF.join(
    new_psd_EC_DUKE_DF,
    on=["# SbjID", "TrialNum"],
    how="inner",
    lsuffix="_EO",
    rsuffix="_EC",
)

new_psd_EO_SAGES_DF.to_csv(
    path_or_buf=os.path.join(csvTestDir, "new_psd_EO_SAGES_DF.csv")
)

new_psd_EC_SAGES_DF.to_csv(
    path_or_buf=os.path.join(csvTestDir, "new_psd_EC_SAGES_DF.csv")
)

new_psd_SAGES_DF.to_csv(path_or_buf=os.path.join(csvTestDir, "new_psd_SAGES_DF.csv"))

new_psd_EO_DUKE_DF.to_csv(
    path_or_buf=os.path.join(csvTestDir, "new_psd_EO_DUKE_DF.csv")
)

new_psd_EC_DUKE_DF.to_csv(
    path_or_buf=os.path.join(csvTestDir, "new_psd_EC_DUKE_DF.csv")
)

new_psd_DUKE_DF.to_csv(path_or_buf=os.path.join(csvTestDir, "new_psd_DUKE_DF.csv"))

# Compute diff and ratios
# Want ratio to be between 0 and 1 so EO/EC because EC is typically larger
# Before distance to the median transformation!!!
new_psd_SAGES_DF["O1_alpha_Ratio"] = (
    new_psd_SAGES_DF["O1_alpha_EO"] / new_psd_SAGES_DF["O1_alpha_EC"]
)
new_psd_SAGES_DF["O2_alpha_Ratio"] = (
    new_psd_SAGES_DF["O2_alpha_EO"] / new_psd_SAGES_DF["O2_alpha_EC"]
)
new_psd_SAGES_DF["POz_alpha_Ratio"] = (
    new_psd_SAGES_DF["POz_alpha_EO"] / new_psd_SAGES_DF["POz_alpha_EC"]
)

new_psd_SAGES_DF["O1_alpha_Diff"] = (
    new_psd_SAGES_DF["O1_alpha_EC"] - new_psd_SAGES_DF["O1_alpha_EO"]
)
new_psd_SAGES_DF["O2_alpha_Diff"] = (
    new_psd_SAGES_DF["O2_alpha_EC"] - new_psd_SAGES_DF["O2_alpha_EO"]
)
new_psd_SAGES_DF["POz_alpha_Diff"] = (
    new_psd_SAGES_DF["POz_alpha_EC"] - new_psd_SAGES_DF["POz_alpha_EO"]
)


new_psd_DUKE_DF["O1_alpha_Ratio"] = (
    new_psd_DUKE_DF["O1_alpha_EO"] / new_psd_DUKE_DF["O1_alpha_EC"]
)
new_psd_DUKE_DF["O2_alpha_Ratio"] = (
    new_psd_DUKE_DF["O2_alpha_EO"] / new_psd_DUKE_DF["O2_alpha_EC"]
)
new_psd_DUKE_DF["POz_alpha_Ratio"] = (
    new_psd_DUKE_DF["POz_alpha_EO"] / new_psd_DUKE_DF["POz_alpha_EC"]
)

new_psd_DUKE_DF["O1_alpha_Diff"] = (
    new_psd_DUKE_DF["O1_alpha_EC"] - new_psd_DUKE_DF["O1_alpha_EO"]
)
new_psd_DUKE_DF["O2_alpha_Diff"] = (
    new_psd_DUKE_DF["O2_alpha_EC"] - new_psd_DUKE_DF["O2_alpha_EO"]
)
new_psd_DUKE_DF["POz_alpha_Diff"] = (
    new_psd_DUKE_DF["POz_alpha_EC"] - new_psd_DUKE_DF["POz_alpha_EO"]
)

# if op_ROI:
#     roi_dict = getROIMd.getROICh(op_10_20)
#     for i_col, (col_name, col_list) in enumerate(roi_dict.items()):
#         spr_df[col_name] = spr_df[col_list].mean(axis=1)
#     spr_df = spr_df[roi_dict.keys()]

# spr_df.to_csv(val_FN, header=True, sep=',')

# Normalize all samples

## Transform EEG features accordingly
## Add pydantic and data validation here.
## Make sure only 1 is NaN for each column
if dist_Medn == 1:
    new_psd_transform_SAGES_DF = new_psd_SAGES_DF.apply(
        lambda x: np.abs((x - x.median()))
    )
    new_psd_transform_DUKE_DF = new_psd_DUKE_DF.apply(
        lambda x: np.abs((x - x.median()))
    )
elif dist_Medn == 2:
    # distance between log10 spr and the log10 of the median
    new_psd_transform_SAGES_DF = new_psd_SAGES_DF.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - np.log10(x.median())) != 0,
            np.abs(np.log10(x) - np.log10(x.median())),
            np.NaN,
        )
    )

    sages_Median = np.log10(new_psd_SAGES_DF.median(axis=0))

    new_psd_transform_DUKE_DF = new_psd_DUKE_DF.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - sages_Median.loc[x.name]) != 0,
            np.abs(np.log10(x) - sages_Median.loc[x.name]),
            np.NaN,
        ),
        axis=0,
    )
elif dist_Medn == 3:
    whole_medn = np.log10(np.median(new_psd_SAGES_DF))
    new_psd_transform_SAGES_DF = new_psd_SAGES_DF.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - whole_medn) != 0,
            np.abs(np.log10(x) - whole_medn),
            np.NaN,
        )
    )
    whole_medn = np.log10(np.median(new_psd_DUKE_DF))
    new_psd_transform_DUKE_DF = new_psd_DUKE_DF.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - whole_medn) != 0,
            np.abs(np.log10(x) - whole_medn),
            np.NaN,
        )
    )
# elif dist_Medn == 4:
#     scaler = MinMaxScaler(feature_range=(-1,1))
#
#     feats_EO_SAGES_DF[numeric_cols] = scaler.fit_transform(
# feats_EO_SAGES_DF[numeric_cols])
#
#     feats_EO_SAGES_DF[numeric_cols] = feats_EO_SAGES_DF[numeric_cols].
# apply(lambda x: np.where(
#         np.abs(np.arcsinh(x, where=(x!=0))-np.arcsinh(x.median()))!=0,
#         np.abs(np.arcsinh(x, where=(x!=0))-np.arcsinh(x.median())),
#         0)
#     )
elif dist_Medn == 6:
    new_psd_transform_SAGES_DF = new_psd_SAGES_DF.apply(lambda x: 10 * np.log10(x))
    new_psd_transform_DUKE_DF = new_psd_DUKE_DF.apply(lambda x: 10 * np.log10(x))
else:
    new_psd_transform_SAGES_DF = new_psd_SAGES_DF.apply(scipy.stats.zscore)
    new_psd_transform_DUKE_DF = new_psd_DUKE_DF.apply(scipy.stats.zscore)

## Replace NaN with 0 for log10(0). Want distance to be 0.
new_psd_transform_SAGES_DF.replace(np.NaN, 0, inplace=True)
new_psd_transform_DUKE_DF.replace(np.NaN, 0, inplace=True)

##
new_psd_transform_SAGES_DF.to_csv(
    path_or_buf=os.path.join(csvTestDir, "new_psd_transform_SAGES_DF.csv")
)

new_psd_transform_DUKE_DF.to_csv(
    path_or_buf=os.path.join(csvTestDir, "new_psd_transform_DUKE_DF.csv")
)

# if op_RemCollinearity:
#     new_psd_transform_SAGES_DF = rem_Multicollinearity.remove_vif(
#    new_psd_transform_SAGES_DF, thresh=40)
#     new_psd_transform_DUKE_DF = rem_Multicollinearity.remove_vif(
#    new_psd_transform_DUKE_DF, thresh=40)

## Add cognitive assessment to dfs.
if op_CogAssmnt:
    new_psd_transform_SAGES_DF[
        "MoCA"
    ] = new_psd_transform_SAGES_DF.index.get_level_values(0).map(cogScore_SAGES_dict)

    mu_SAGES = new_psd_transform_SAGES_DF["MoCA"].mean()
    # ddof = 0
    sigma_SAGES = new_psd_transform_SAGES_DF["MoCA"].std(ddof=0)

    if op_Scl_CogAssmnt == 0 or op_Scl_CogAssmnt == 3 or op_Scl_CogAssmnt == 4:
        new_psd_transform_SAGES_DF["MoCA"] = (
            new_psd_transform_SAGES_DF["MoCA"]
            .to_frame()
            .apply(zscore, nan_policy="omit")
        )
    elif op_Scl_CogAssmnt == 2:
        new_psd_transform_SAGES_DF["MoCA"] = (
            new_psd_transform_SAGES_DF["MoCA"]
            .to_frame()
            .apply(
                lambda x: np.where(
                    np.abs(np.log10(x) - np.log10(x.median())) != 0,
                    np.abs(np.log10(x) - np.log10(x.median())),
                    np.NaN,
                )
            )
        )
    else:
        pass

    new_psd_transform_DUKE_DF[
        "MoCA"
    ] = new_psd_transform_DUKE_DF.index.get_level_values(0).map(cogScore_DUKE_dict)
    if op_Scl_CogAssmnt == 0:
        new_psd_transform_DUKE_DF["MoCA"] = (
            new_psd_transform_DUKE_DF["MoCA"]
            .to_frame()
            .apply(zscore, nan_policy="omit")
        )
    elif op_Scl_CogAssmnt == 4:
        new_psd_transform_DUKE_DF["MoCA"] = (
            new_psd_transform_DUKE_DF["MoCA"]
            .to_frame()
            .apply(lambda x: (x - mu_SAGES) / sigma_SAGES)
        )
    elif op_Scl_CogAssmnt == 2:
        new_psd_transform_DUKE_DF["MoCA"] = (
            new_psd_transform_DUKE_DF["MoCA"]
            .to_frame()
            .apply(
                lambda x: np.where(
                    np.abs(np.log10(x) - np.log10(x.median())) != 0,
                    np.abs(np.log10(x) - np.log10(x.median())),
                    np.NaN,
                )
            )
        )
    # PMCID: PMC5545909
    # PMID: 28697562
    elif op_Scl_CogAssmnt == 3:
        n_tot_SWEDEN = 758 + 102
        mu_SWEDEN = (26 * 758 + 21.6 * 102) / (758 + 102)
        q1 = (2.3**2) * 758 + 758 * (26**2)
        q2 = (4.3**2) * 102 + 102 * (21.6**2)
        qc = q1 + q2
        std_SWEDEN = ((qc - (758 + 102) * mu_SWEDEN**2) / (758 + 102)) ** (1 / 2)
        new_psd_transform_DUKE_DF["MoCA"] = (
            new_psd_transform_DUKE_DF["MoCA"]
            .to_frame()
            .apply(lambda x: (x - mu_SWEDEN) / std_SWEDEN)
        )
    else:
        pass

## Add sex to dataframes.
if op_Sex:
    new_psd_transform_SAGES_DF[
        "Sex"
    ] = new_psd_transform_SAGES_DF.index.get_level_values(0).map(sex_dict_SAGES)
    new_psd_transform_DUKE_DF["Sex"] = new_psd_transform_DUKE_DF.index.get_level_values(
        0
    ).map(sex_dict_DUKE)

## Add age to dataframes.
if op_Age:
    new_psd_transform_SAGES_DF[
        "Age"
    ] = new_psd_transform_SAGES_DF.index.get_level_values(0).map(age_dict_SAGES)
    new_psd_transform_SAGES_DF["Age"] = (
        new_psd_transform_SAGES_DF["Age"].to_frame().apply(zscore, nan_policy="omit")
    )

    new_psd_transform_DUKE_DF["Age"] = new_psd_transform_DUKE_DF.index.get_level_values(
        0
    ).map(age_dict_DUKE)
    new_psd_transform_DUKE_DF["Age"] = (
        new_psd_transform_DUKE_DF["Age"].to_frame().apply(zscore, nan_policy="omit")
    )

## Filter to low MoCA. Not needed.
if op_LowMoCA:
    lowMoCA_List = [k for k, v in cogScore_SAGES_dict.items() if v < 21]

    new_psd_transform_SAGES_DF = new_psd_transform_SAGES_DF.loc[
        new_psd_transform_SAGES_DF.index.get_level_values(0).isin(lowMoCA_List)
    ]

    lowMoCA_List = [k for k, v in cogScore_DUKE_dict.items() if v < 21]

    new_psd_transform_DUKE_DF = new_psd_transform_DUKE_DF.loc[
        new_psd_transform_DUKE_DF.index.get_level_values(0).isin(lowMoCA_List)
    ]

# new_psd_EO_SAGES_DF.drop(['# SbjID', 'TrialNum',
#'SetFile'],axis=1,inplace=True)
# new_psd_EO_SAGES_DF.replace(np.NaN,0,inplace=True)
# scikit learn does not use indices as features but just to be extra safe...


# # define groups of columns for group lasso for logistic regression
# list_regex = []
# for col_name, _ in new_psd_EO_SAGES_DF.items():
#     if col_name not in ['# SbjID', 'TrialNum','SetFile','MoCA']:
#         temp = col_name.rsplit('_',maxsplit=1)[0]
#         if temp not in list_regex:
#             list_regex.append(temp)
#
# groups = np.zeros(new_psd_EO_SAGES_DF.shape[1])
#
# for i_ptrn, regex_ptrn in enumerate(list_regex):
#     temp02 = list(new_psd_EO_SAGES_DF.filter(regex=regex_ptrn))
#     id_drop_chn = [new_psd_EO_SAGES_DF.columns
#     .get_loc(col_name) for col_name in temp02]
#     groups[id_drop_chn] = i_ptrn+1
#
# params = dict()
# params['groups'] = groups

## Get delirium status.
CAM_SAGES = getDeliriumStatusMd.getDeliriumStatus(1)
CAM_DUKE = getDeliriumStatusMd.getDeliriumStatus(0)

## Encode delirium status. 1/positive for having delirium.
responses_SAGES = new_psd_transform_SAGES_DF.index.get_level_values(0).to_frame()
responses_DUKE = new_psd_transform_DUKE_DF.index.get_level_values(0).to_frame()
# try {0,1} and {-1,1} encoding. will get different results.
responses_SAGES["Y"] = np.where(responses_SAGES["# SbjID"].isin(CAM_SAGES), 1, 0)
responses_SAGES = responses_SAGES["Y"]
responses_DUKE["Y"] = np.where(responses_DUKE["# SbjID"].isin(CAM_DUKE), 1, 0)
responses_DUKE = responses_DUKE["Y"]

# Does NOT support group as advertised!!!
# It's actually a scaling coefficient alpha that weight
# between L1 and L2 norm.
# https://github.com/scikit-learn-contrib/lightning

## List of frequency bands.
# band_names = ["delta", "theta", "alpha", "beta"]
band_names = ["alpha"]
# band_names = ["theta"]

# # manual
# #manual_feat_selection = getManFeatsFn()
# if op_10_20:
#     if binWidth == 2 and not bin2_startF:
#         manual_feat_selection = ['O1_10_EO',
#                                  'O2_10_EO',
#                                  'POz_10_EO',
#                                  'O1_8_EC',
#                                  'O2_8_EC',
#                                  'POz_8_EC']
#     else:
#         manual_feat_selection = ['O1_11_EO',
#                                  'O2_11_EO',
#                                  'POz_11_EO',
#                                  'O1_9_EC',
#                                  'O2_9_EC',
#                                  'POz_9_EC']
# else:
#     if binWidth == 2 and not bin2_startF:
#         manual_feat_selection = ['O1_10_EO',
#                                  'O2_10_EO',
#                                  'Oz_10_EO',
#                                  'POz_10_EO',
#                                  'PO3_10_EO',
#                                  'PO4_10_EO',
#                                  'O1_8_EC',
#                                  'O2_8_EC',
#                                  'Oz_8_EC',
#                                  'POz_8_EC',
#                                  'PO3_8_EC',
#                                  'PO4_8_EC']
#     else:
#         manual_feat_selection = ['O1_11_EO',
#                                  'O2_11_EO',
#                                  'Oz_11_EO',
#                                  'POz_11_EO',
#                                  'PO3_11_EO',
#                                  'PO4_11_EO',
#                                  'O1_9_EC',
#                                  'O2_9_EC',
#                                  'Oz_9_EC',
#                                  'POz_9_EC',
#                                  'PO3_9_EC',
#                                  'PO4_9_EC']

## Get list of channels in Duke's 10-20 montage.
chns_10_20 = get1020Md.get1020()
chns_10_20.append("MoCA")

## Delete rows with missing scores from Duke dataframe and responses.
# rows_to_del = responses_DUKE[responses_DUKE.index.isin(["0190", "046", "025","027","051","094","185"])]
rows_to_del = responses_DUKE[responses_DUKE.index.isin(["0190", "046"])]
new_psd_transform_DUKE_DF.drop(rows_to_del.index, axis=0, inplace=True)
responses_DUKE.drop(rows_to_del.index, inplace=True)

## Get dictionary of classifiers to try.
if op_Ensemble:
    mdlDict = getEnsembleModelFn()
else:
    mdlDict = getModelCVFn()
    # mdlDict = getModelNSCFn()

## Compute degree of freedom for confidence intervals.
dof_SAGES = new_psd_transform_SAGES_DF.shape[0] - 2
dof_DUKE = new_psd_transform_DUKE_DF.shape[0] - 2

## Define dictionary of different performance metrics.
scoring = {
    "accuracy": make_scorer(metrics.accuracy_score),
    "sensitivity": make_scorer(metrics.recall_score),
    "specificity": make_scorer(metrics.recall_score, pos_label=0),
    "f1": make_scorer(metrics.f1_score),
    "roc_auc": make_scorer(metrics.roc_auc_score),
    "pr_auc": make_scorer(metrics.average_precision_score),
    "PPV": make_scorer(metrics.precision_score),
    "NPV": make_scorer(metrics.precision_score, pos_label=0),
}

## Define cross-validation.
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

## Loop through frequency bands.
for band_i, band_name in enumerate(band_names):
    if op_10_20:
        # manual_feat_selection = [
        #     f"O1_{band_name}_EC",
        #     f"O2_{band_name}_EC",
        #     f"POz_{band_name}_EC",
        #     f"P7_{band_name}_EC",
        #     f"P8_{band_name}_EC",
        #     f"CP5_{band_name}_EC",
        #     f"CP6_{band_name}_EC",
        #     f"F4_{band_name}_EC",
        #     f"F3_{band_name}_EC",
        #     f"Fp1_{band_name}_EC",
        #     f"Fp2_{band_name}_EC",
        # ]

        manual_feat_selection = [
            f"O1_{band_name}_EO",
            f"O2_{band_name}_EO",
            f"POz_{band_name}_EO",
            f"O1_{band_name}_EC",
            f"O2_{band_name}_EC",
            f"POz_{band_name}_EC",
        ]

        # manual_feat_selection = [
        #     f"O1_{band_name}_EO",
        #     f"O2_{band_name}_EO",
        #     f"O1_{band_name}_EC",
        #     f"O2_{band_name}_EC",
        #     f"O1_{band_name}_Ratio",
        #     f"O2_{band_name}_Ratio",
        # ]

        # manual_feat_selection = [
        #     f"O1_{band_name}_EO",
        #     f"O2_{band_name}_EO",
        #     f"POz_{band_name}_EO",
        #     f"O1_{band_name}_EC",
        #     f"O2_{band_name}_EC",
        #     f"POz_{band_name}_EC",
        #     f"O1_{band_name}_Ratio",
        #     f"O2_{band_name}_Ratio",
        #     f"POz_{band_name}_Ratio",
        # ]

    else:
        manual_feat_selection = [
            f"O1_{band_name}_EO",
            f"O2_{band_name}_EO",
            f"Oz_{band_name}_EO",
            f"POz_{band_name}_EO",
            f"PO3_{band_name}_EO",
            f"PO4_{band_name}_EO",
            f"O1_{band_name}_EC",
            f"O2_{band_name}_EC",
            f"Oz_{band_name}_EC",
            f"POz_{band_name}_EC",
            f"PO3_{band_name}_EC",
            f"PO4_{band_name}_EC",
        ]

    if op_FeatSelect:
        feat_sel_SAGES_FN = os.path.join(
            figDir, f"featList_BP_{band_name}_{fn_opt}.txt"
        )

    mdl_Scores: Dict[str, Any] = dict.fromkeys(mdlDict.keys(), 0)
    mdl_Acc = np.zeros((len(mdlDict)))
    mdl_F1 = np.zeros((len(mdlDict)))
    mdl_ROC_AUC = np.zeros((len(mdlDict)))
    mdl_sensitivity = np.zeros((len(mdlDict)))
    mdl_specificity = np.zeros((len(mdlDict)))
    mdl_PR_AUC = np.zeros((len(mdlDict)))
    mdl_PPV = np.zeros((len(mdlDict)))
    mdl_Acc_CI = np.zeros((len(mdlDict)))
    mdl_NPV = np.zeros((len(mdlDict)))

    # training perf
    mdl_Acc_tr = np.zeros((len(mdlDict)))
    mdl_F1_tr = np.zeros((len(mdlDict)))
    mdl_ROC_AUC_tr = np.zeros((len(mdlDict)))
    mdl_sensitivity_tr = np.zeros((len(mdlDict)))
    mdl_specificity_tr = np.zeros((len(mdlDict)))
    mdl_PR_AUC_tr = np.zeros((len(mdlDict)))
    mdl_PPV_tr = np.zeros((len(mdlDict)))
    mdl_Acc_CI_tr = np.zeros((len(mdlDict)))
    mdl_NPV_tr = np.zeros((len(mdlDict)))

    ## Filter by band name first.
    regex_str = f"(_{band_name})"

    if op_CogAssmnt:
        regex_str += "|(MoCA)"
        manual_feat_selection.append("MoCA")
    if op_Sex:
        regex_str += "|(Sex)"
        manual_feat_selection.append("Sex")
    if op_Age:
        regex_str += "|(Age)"
        manual_feat_selection.append("Age")

    fBand_SAGES_DF = new_psd_transform_SAGES_DF.filter(regex=regex_str, axis="columns")
    fBand_DUKE_DF = new_psd_transform_DUKE_DF.filter(regex=regex_str, axis="columns")

    ## Manual feature selection on dataframes
    if op_ManualFeat:
        fBand_SAGES_DF = fBand_SAGES_DF[manual_feat_selection]
        fBand_DUKE_DF = fBand_DUKE_DF[manual_feat_selection]

    if op_10_20 and not op_ManualFeat and not op_ROI:
        fBand_SAGES_DF = fBand_SAGES_DF[chns_10_20]
        fBand_DUKE_DF = fBand_DUKE_DF[chns_10_20]

    ## Convert dataframes to numpy arrays since I'm paranoid that indices
    ## may be used during classification despite developers' assurance that
    ## indices aren't used at all.
    predictors_Arr_SAGES = fBand_SAGES_DF.to_numpy()
    predictors_Arr_DUKE = fBand_DUKE_DF.to_numpy()

    ## Concatenate SAGES and Duke arrays
    predictors_Arr = np.concatenate((predictors_Arr_SAGES, predictors_Arr_DUKE), axis=0)
    responses = np.concatenate((responses_SAGES, responses_DUKE), axis=0)

    np.savetxt(
        os.path.join(csvTestDir, "predictors_Arr.csv"), predictors_Arr, delimiter=","
    )
    np.savetxt(os.path.join(csvTestDir, "responses.csv"), responses, delimiter=",")

    ## Get training and test indices for concatenated arrays.
    if op_Training == "SAGES" or op_Training == "BSErr":
        train_idx = [i for i in range(predictors_Arr_SAGES.shape[0])]
        test_idx = [
            i
            for i in range(
                predictors_Arr_SAGES.shape[0],
                predictors_Arr_SAGES.shape[0] + predictors_Arr_DUKE.shape[0],
            )
        ]
    elif op_Training == "DUKE":
        train_idx = [
            i
            for i in range(
                predictors_Arr_SAGES.shape[0],
                predictors_Arr_SAGES.shape[0] + predictors_Arr_DUKE.shape[0],
            )
        ]
        test_idx = [i for i in range(predictors_Arr_SAGES.shape[0])]

    ## Open text file to list features selected if using op_FeatSelect
    f_feat: Union[TextIO, ContextManager[None]] = (
        open(feat_sel_SAGES_FN, "w")
        if op_FeatSelect
        else nullcontext()
        # open(featFN, "w", encoding="utf-8") if op_FeatSelect else nullcontext()
    )

    if f_feat is None:
        raise RuntimeError(f"Failed to open {feat_sel_SAGES_FN}")
    # assert isinstance(f_feat, TextIO)
    # if isinstance(f_feat, TextIO):
    # else:
    #     assert_never(f_feat)

    with f_feat:
        if op_Training == "SAGES" or op_Training == "DUKE":
            ## Loop through different classifiers.
            for i, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
                print(mdl_name)
                if isinstance(f_feat, TextIO):
                    ## Write model name to feature selection text file.
                    f_feat.write(f"{mdl_name}\n")
                if op_FeatSelect == 1:
                    pipeline = Pipeline(
                        [
                            ("f_selection", SelectKBest(f_classif, k=num_F)),
                            ("classifier", mdl_instance),
                        ]
                    )
                    mdl_Scores[mdl_name] = _fit_and_score(
                        pipeline,
                        predictors_Arr,
                        responses,
                        scoring,
                        train_idx,
                        test_idx,
                        False,
                        None,
                        None,
                        error_score="raise",
                        return_estimator=True,
                    )
                elif op_FeatSelect == 2:
                    pipeline = Pipeline(
                        [
                            ("f_selection", SelectKBest(mutual_info_classif, k=num_F)),
                            ("classifier", mdl_instance),
                        ]
                    )
                    mdl_Scores[mdl_name] = _fit_and_score(
                        pipeline,
                        predictors_Arr,
                        responses,
                        scoring,
                        train_idx,
                        test_idx,
                        False,
                        None,
                        None,
                        error_score="raise",
                        return_estimator=True,
                    )
                elif op_FeatSelect == 3:
                    # ridge = RidgeCV(alphas=np.logspace(-6,6,num=13))
                    ridge = RidgeClassifier(alpha=1)
                    pipeline = Pipeline(
                        [
                            (
                                "f_selection",
                                SequentialFeatureSelector(
                                    ridge,
                                    n_features_to_select=num_F,
                                    direction="forward",
                                ),
                            ),
                            ("classifier", mdl_instance),
                        ]
                    )
                    mdl_Scores[mdl_name] = _fit_and_score(
                        pipeline,
                        predictors_Arr,
                        responses,
                        scoring,
                        train_idx,
                        test_idx,
                        False,
                        None,
                        None,
                        error_score="raise",
                        return_estimator=True,
                    )
                elif op_FeatSelect == 4:
                    # ridge = RidgeCV(alphas=np.logspace(-6, 6, num=13))
                    nsc = _NSC.NearestCentroid(metric="manhattan", shrink_threshold=0.5)
                    pipeline = Pipeline(
                        [
                            (
                                "f_selection",
                                SequentialFeatureSelector(
                                    nsc, n_features_to_select=num_F, direction="forward"
                                ),
                            ),
                            ("classifier", mdl_instance),
                        ]
                    )
                    mdl_Scores[mdl_name] = _fit_and_score(
                        pipeline,
                        predictors_Arr,
                        responses,
                        scoring,
                        train_idx,
                        test_idx,
                        False,
                        None,
                        None,
                        error_score="raise",
                        return_estimator=True,
                    )
                else:
                    mdl_Scores[mdl_name] = _fit_and_score(
                        mdl_instance,
                        predictors_Arr,
                        responses,
                        scoring,
                        train_idx,
                        test_idx,
                        False,
                        None,
                        None,
                        error_score="raise",
                        return_estimator=True,
                    )

                ## Save scores to variables.
                temp_mdl = mdl_Scores[mdl_name]
                temp_scores = mdl_Scores[mdl_name]["test_scores"]["accuracy"]
                mdl_Acc[i] = np.mean(temp_scores)
                mdl_ROC_AUC[i] = np.mean(mdl_Scores[mdl_name]["test_scores"]["roc_auc"])
                mdl_PR_AUC[i] = np.mean(mdl_Scores[mdl_name]["test_scores"]["pr_auc"])

                mdl_sensitivity[i] = np.mean(
                    mdl_Scores[mdl_name]["test_scores"]["sensitivity"]
                )
                mdl_specificity[i] = np.mean(
                    mdl_Scores[mdl_name]["test_scores"]["specificity"]
                )
                mdl_F1[i] = 2 / (mdl_PPV[i] ** (-1) + mdl_sensitivity[i] ** (-1))
                mdl_PPV[i] = np.mean(mdl_Scores[mdl_name]["test_scores"]["PPV"])
                mdl_NPV = np.mean(mdl_Scores[mdl_name]["test_scores"]["NPV"])

                ## Write feature names to feature selection text file.
                if isinstance(f_feat, TextIO):
                    f_feat.write("Feature Selection")
                    for el in temp_mdl["estimator"][0].get_feature_names_out(
                        fBand_SAGES_DF.columns
                    ):
                        f_feat.write(f"{el}\t")
                    f_feat.write("\n\n")

                ## Write model instances' coefficients to csv file.
                if not op_FeatSelect and hasattr(temp_mdl["estimator"], "coef_"):
                    coefFN = os.path.join(
                        figDir, f"coeff_{band_name}_{mdl_name}_{fn_opt}.csv"
                    )
                    with open(coefFN, "w") as f_coeff:
                        for col_name in fBand_SAGES_DF.columns:
                            f_coeff.write(f"{col_name},")
                        f_coeff.write("\n")
                        temp_coeff = temp_mdl["estimator"].coef_
                        for i_coeff in range(temp_coeff.shape[1]):
                            f_coeff.write(f"{temp_coeff[0,i_coeff]},")
                        f_coeff.write("\n")

            err_df = pd.DataFrame(
                {
                    "Accuracy": mdl_Acc_CI,
                    "Sensitivity": np.zeros(len(mdlDict)),
                    "Specificity": np.zeros(len(mdlDict)),
                    "f1": np.zeros(len(mdlDict)),
                    "roc_auc": np.zeros(len(mdlDict)),
                    "pr_auc": np.zeros(len(mdlDict)),
                    "precision": np.zeros(len(mdlDict)),
                }
            )

            scores_df = pd.DataFrame(
                {
                    "Model": mdlDict.keys(),
                    "Accuracy": mdl_Acc,
                    "Sensitivity": mdl_sensitivity,
                    "Specificity": mdl_specificity,
                    "f1": mdl_F1,
                    "roc_auc": mdl_ROC_AUC,
                    "pr_auc": mdl_PR_AUC,
                    "precision": mdl_PPV,
                }
            )
        elif op_Training == "BS":
            # Here, oversample the minority class before bootstrapping
            # you want to resample from the same original sample space

            # https://wiki.python.org/moin/TimeComplexity
            # 5000 repetitions
            # 85 samples

            # mdl_Scores: Dict[str, Any] = dict.fromkeys(mdlDict.keys(), 0)
            mdl_Acc_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_F1_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_ROC_AUC_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_sensitivity_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_specificity_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_PR_AUC_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_PPV_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_NPV_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdlAccCI_bs = np.zeros((len(mdlDict), num_rep_bs))

            ## Actually not needed.
            dof_bs = num_rep_bs - 1

            ## Perform over-sampling on the minority class before bootstrapping
            if op_Oversampling == 1:
                ros = RandomOverSampler(
                    random_state=0, sampling_strategy=0.6, shrinkage=1
                )
                predictors_Arr_SAGES_os, responses_SAGES_os = ros.fit_resample(
                    predictors_Arr_SAGES,
                    responses_SAGES,
                )
            elif op_Oversampling == 2:
                sm = SMOTE(random_state=0, sampling_strategy=0.6)
                predictors_Arr_SAGES_os, responses_SAGES_os = sm.fit_resample(
                    predictors_Arr_SAGES,
                    responses_SAGES,
                )
            else:
                predictors_Arr_SAGES_os = predictors_Arr_SAGES
                responses_SAGES_os = responses_SAGES

            ## Bootstrap iteration.
            for iter_bs in range(num_rep_bs):
                pred_SAGES_bs, resp_SAGES_bs = resample(
                    predictors_Arr_SAGES_os,
                    responses_SAGES_os,
                    n_samples=predictors_Arr_SAGES_os.shape[0],
                    stratify=responses_SAGES_os,
                )

                ## Combine training and test sets into one numpy arrays.
                predictors_Arr = np.concatenate(
                    (pred_SAGES_bs, predictors_Arr_DUKE), axis=0
                )
                responses = np.concatenate((resp_SAGES_bs, responses_DUKE), axis=0)

                ## Get indices of training and test points in combined arrays.
                train_idx = [i for i in range(pred_SAGES_bs.shape[0])]
                test_idx = [
                    i
                    for i in range(
                        pred_SAGES_bs.shape[0],
                        pred_SAGES_bs.shape[0] + predictors_Arr_DUKE.shape[0],
                    )
                ]

                mdl_temp_Scores: Dict[str, Any] = dict.fromkeys(mdlDict.keys(), 0)
                ## Iterate over different classifiers.
                for i_mdl, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
                    if op_FeatSelect == 1:
                        pipeline = Pipeline(
                            [
                                ("f_selection", SelectKBest(f_classif, k=num_F)),
                                ("classifier", mdl_instance),
                            ]
                        )
                        mdl_temp_Scores[mdl_name] = _fit_and_score(
                            pipeline,
                            predictors_Arr,
                            responses,
                            scoring,
                            train_idx,
                            test_idx,
                            False,
                            None,
                            None,
                            error_score="raise",
                            return_estimator=True,
                        )
                    elif op_FeatSelect == 2:
                        pipeline = Pipeline(
                            [
                                (
                                    "f_selection",
                                    SelectKBest(mutual_info_classif, k=num_F),
                                ),
                                ("classifier", mdl_instance),
                            ]
                        )
                        mdl_temp_Scores[mdl_name] = _fit_and_score(
                            pipeline,
                            predictors_Arr,
                            responses,
                            scoring,
                            train_idx,
                            test_idx,
                            False,
                            None,
                            None,
                            error_score="raise",
                            return_estimator=True,
                        )
                    elif op_FeatSelect == 3:
                        # ridge = RidgeCV(alphas=np.logspace(-6,6,num=13))
                        ridge = RidgeClassifier(alpha=1)
                        pipeline = Pipeline(
                            [
                                (
                                    "f_selection",
                                    SequentialFeatureSelector(
                                        ridge,
                                        n_features_to_select=num_F,
                                        direction="forward",
                                    ),
                                ),
                                ("classifier", mdl_instance),
                            ]
                        )
                        mdl_temp_Scores[mdl_name] = _fit_and_score(
                            pipeline,
                            predictors_Arr,
                            responses,
                            scoring,
                            train_idx,
                            test_idx,
                            False,
                            None,
                            None,
                            error_score="raise",
                            return_estimator=True,
                        )
                    elif op_FeatSelect == 4:
                        nsc = _NSC.NearestCentroid(
                            metric="manhattan", shrink_threshold=0.5
                        )
                        pipeline = Pipeline(
                            [
                                (
                                    "f_selection",
                                    SequentialFeatureSelector(
                                        nsc,
                                        n_features_to_select=num_F,
                                        direction="forward",
                                    ),
                                ),
                                ("classifier", mdl_instance),
                            ]
                        )
                        mdl_temp_Scores[mdl_name] = _fit_and_score(
                            pipeline,
                            predictors_Arr,
                            responses,
                            scoring,
                            train_idx,
                            test_idx,
                            False,
                            None,
                            None,
                            error_score="raise",
                            return_estimator=True,
                        )
                    else:
                        mdl_temp_Scores[mdl_name] = _fit_and_score(
                            mdl_instance,
                            predictors_Arr,
                            responses,
                            scoring,
                            train_idx,
                            test_idx,
                            False,
                            None,
                            None,
                            error_score="raise",
                            return_estimator=True,
                        )

                    # unknown mean and std
                    mdl_Acc_bs[i_mdl, iter_bs] = mdl_temp_Scores[mdl_name][
                        "test_scores"
                    ]["accuracy"]
                    mdl_F1_bs[i_mdl, iter_bs] = mdl_temp_Scores[mdl_name][
                        "test_scores"
                    ]["f1"]
                    mdl_ROC_AUC_bs[i_mdl, iter_bs] = mdl_temp_Scores[mdl_name][
                        "test_scores"
                    ]["roc_auc"]
                    mdl_PR_AUC_bs[i_mdl, iter_bs] = mdl_temp_Scores[mdl_name][
                        "test_scores"
                    ]["pr_auc"]
                    mdl_PPV_bs[i_mdl, iter_bs] = mdl_temp_Scores[mdl_name][
                        "test_scores"
                    ]["PPV"]
                    mdl_NPV_bs[i_mdl, iter_bs] = mdl_temp_Scores[mdl_name][
                        "test_scores"
                    ]["NPV"]
                    mdl_sensitivity_bs[i_mdl, iter_bs] = mdl_temp_Scores[mdl_name][
                        "test_scores"
                    ]["sensitivity"]
                    mdl_specificity_bs[i_mdl, iter_bs] = mdl_temp_Scores[mdl_name][
                        "test_scores"
                    ]["specificity"]
            # check dimension here, that mean is looking at the right axis
            mdl_Acc = np.mean(mdl_Acc_bs, axis=1)
            # cannot import unittest.TestCase
            # tc = TestCase()
            # tc.assertTupleEqual(mdl_Acc.shape, (len(mdlDict)))
            ## Get the mean of different scores of bootstrapped samples.
            mdl_ROC_AUC = np.mean(mdl_ROC_AUC_bs, axis=1)
            mdl_PR_AUC = np.mean(mdl_PR_AUC_bs, axis=1)
            mdl_PPV = np.mean(mdl_PPV_bs, axis=1)
            mdl_NPV = np.mean(mdl_NPV_bs, axis=1)
            mdl_sensitivity = np.mean(mdl_sensitivity_bs, axis=1)
            mdl_specificity = np.mean(mdl_specificity_bs, axis=1)
            mdl_F1 = 2 / (mdl_PPV ** (-1) + mdl_sensitivity ** (-1))

            ## Get the median of different scores of bootstrapped samples.
            mdl_Acc_med = np.median(mdl_Acc_bs, axis=1)
            mdl_ROC_AUC_med = np.median(mdl_ROC_AUC_bs, axis=1)
            mdl_PR_AUC_med = np.median(mdl_PR_AUC_bs, axis=1)
            mdl_PPV_med = np.median(mdl_PPV_bs, axis=1)
            mdl_NPV_med = np.median(mdl_NPV_bs, axis=1)
            mdl_sensitivity_med = np.median(mdl_sensitivity_bs, axis=1)
            mdl_specificity_med = np.median(mdl_specificity_bs, axis=1)
            mdl_F1_med = 2 / (mdl_PPV_med ** (-1) + mdl_sensitivity_med ** (-1))

            ## Use empirical quantile to compute the 95% confidence intervals.
            mdl_Acc_CI = np.percentile(mdl_Acc_bs, q=[2.5, 97.5], axis=1)
            mdl_F1_CI = np.percentile(mdl_F1_bs, q=[2.5, 97.5], axis=1)
            mdl_ROC_AUC_CI = np.percentile(mdl_ROC_AUC_bs, q=[2.5, 97.5], axis=1)
            mdl_PR_AUC_CI = np.percentile(mdl_PR_AUC_bs, q=[2.5, 97.5], axis=1)
            mdl_PPV_CI = np.percentile(mdl_PPV_bs, q=[2.5, 97.5], axis=1)
            mdl_NPV_CI = np.percentile(mdl_NPV_bs, q=[2.5, 97.5], axis=1)
            mdl_sensitivity_CI = np.percentile(
                mdl_sensitivity_bs, q=[2.5, 97.5], axis=1
            )
            mdl_specificity_CI = np.percentile(
                mdl_specificity_bs, q=[2.5, 97.5], axis=1
            )

            ## Plot histogram of bootstrapped performances.
            fig, axs = plt.subplots(2, 4, figsize=(16, 9.5))
            axs[0, 0].hist(mdl_Acc_bs[1, :])
            axs[0, 0].set_title("Accuracy")

            axs[0, 1].hist(mdl_sensitivity_bs[1, :])
            axs[0, 1].set_title("Sensitivity")

            axs[0, 2].hist(mdl_specificity_bs[1, :])
            axs[0, 2].set_title("Specificity")

            axs[0, 3].hist(mdl_F1_bs[1, :])
            axs[0, 3].set_title("F1")

            axs[1, 0].hist(mdl_ROC_AUC_bs[1, :])
            axs[1, 0].set_title("AUC ROC")

            axs[1, 1].hist(mdl_PR_AUC_bs[1, :])
            axs[1, 1].set_title("AUC PR Curve")

            axs[1, 2].hist(mdl_PPV_bs[1, :])
            axs[1, 2].set_title("PPV")

            axs[1, 3].hist(mdl_NPV_bs[1, :])
            axs[1, 3].set_title("NPV")

            fig.suptitle("LDA CV. Alpha Band.")
            fig.tight_layout()
            bs_lda_cv_fn = os.path.join(figDir, (f"Bootstrap_Hist_LDA_CV.png"))
            fig.savefig(bs_lda_cv_fn)

            fig, axs = plt.subplots(2, 4, figsize=(16, 9.5))
            axs[0, 0].hist(mdl_Acc_bs[2, :])
            axs[0, 0].set_title("Accuracy")

            axs[0, 1].hist(mdl_sensitivity_bs[2, :])
            axs[0, 1].set_title("Sensitivity")

            axs[0, 2].hist(mdl_specificity_bs[2, :])
            axs[0, 2].set_title("Specificity")

            axs[0, 3].hist(mdl_F1_bs[2, :])
            axs[0, 3].set_title("F1")

            axs[1, 0].hist(mdl_ROC_AUC_bs[2, :])
            axs[1, 0].set_title("AUC ROC")

            axs[1, 1].hist(mdl_PR_AUC_bs[2, :])
            axs[1, 1].set_title("AUC PR Curve")

            axs[1, 2].hist(mdl_PPV_bs[2, :])
            axs[1, 2].set_title("PPV")

            axs[1, 3].hist(mdl_NPV_bs[2, :])
            axs[1, 3].set_title("NPV")

            fig.suptitle("LDA LW. Alpha Band.")
            fig.tight_layout()
            bs_lda_lw_fn = os.path.join(figDir, (f"Bootstrap_Hist_LDA_LW.png"))
            fig.savefig(bs_lda_lw_fn)

            fig, axs = plt.subplots(2, 4, figsize=(16, 9.5))
            axs[0, 0].hist(mdl_Acc_bs[4, :])
            axs[0, 0].set_title("Accuracy")

            axs[0, 1].hist(mdl_sensitivity_bs[4, :])
            axs[0, 1].set_title("Sensitivity")

            axs[0, 2].hist(mdl_specificity_bs[4, :])
            axs[0, 2].set_title("Specificity")

            axs[0, 3].hist(mdl_F1_bs[4, :])
            axs[0, 3].set_title("F1")

            axs[1, 0].hist(mdl_ROC_AUC_bs[4, :])
            axs[1, 0].set_title("AUC ROC")

            axs[1, 1].hist(mdl_PR_AUC_bs[4, :])
            axs[1, 1].set_title("AUC PR Curve")

            axs[1, 2].hist(mdl_PPV_bs[4, :])
            axs[1, 2].set_title("PPV")

            axs[1, 3].hist(mdl_NPV_bs[4, :])
            axs[1, 3].set_title("NPV")

            fig.suptitle("NSC MH. Alpha Band.")
            fig.tight_layout()
            bs_nsc_mh_fn = os.path.join(figDir, (f"Bootstrap_Hist_NSC_MH.png"))
            fig.savefig(bs_nsc_mh_fn)

            # mdl_Acc_CI = (
            #         t.ppf(1 - (1 - 0.95) / 2, dof_bs)
            #         * np.std(mdl_Acc_bs, axis=1)
            #         / np.sqrt(num_rep_bs)
            #     )
            # #assertTupleEqual(mdl_Acc_CI.shape, (len(mdlDict)))
            # mdl_F1_CI = (
            #         t.ppf(1 - (1 - 0.95) / 2, dof_bs)
            #         * np.std(mdl_F1_bs, axis=1)
            #         / np.sqrt(num_rep_bs)
            #     )
            # mdl_ROC_AUC_CI = (
            #         t.ppf(1 - (1 - 0.95) / 2, dof_bs)
            #         * np.std(mdl_ROC_AUC_bs, axis=1)
            #         / np.sqrt(num_rep_bs)
            #     )
            # mdl_PR_AUC_CI = (
            #         t.ppf(1 - (1 - 0.95) / 2, dof_bs)
            #         * np.std(mdl_PR_AUC_bs, axis=1)
            #         / np.sqrt(num_rep_bs)
            #     )
            # mdl_PPV_CI = (
            #         t.ppf(1 - (1 - 0.95) / 2, dof_bs)
            #         * np.std(mdl_PPV_bs, axis=1)
            #         / np.sqrt(num_rep_bs)
            #     )
            # mdl_sensitivity_CI = (
            #         t.ppf(1 - (1 - 0.95) / 2, dof_bs)
            #         * np.std(mdl_sensitivity_bs, axis=1)
            #         / np.sqrt(num_rep_bs)
            #     )
            # mdl_specificity_CI = (
            #         t.ppf(1 - (1 - 0.95) / 2, dof_bs)
            #         * np.std(mdl_specificity_bs, axis=1)
            #         / np.sqrt(num_rep_bs)
            #     )

            # err_df = pd.DataFrame(
            #     {
            #         "Accuracy": mdl_Acc_CI,
            #         "Sensitivity": mdl_F1_CI,
            #         "Specificity": mdl_ROC_AUC_CI,
            #         "f1": mdl_PR_AUC_CI,
            #         "roc_auc": mdl_PPV_CI,
            #         "pr_auc": mdl_sensitivity_CI,
            #         "precision": mdl_specificity_CI,
            #     }
            # )

            err_df = np.stack(
                (
                    np.abs(mdl_Acc - mdl_Acc_CI),
                    np.abs(mdl_sensitivity - mdl_sensitivity_CI),
                    np.abs(mdl_specificity - mdl_specificity_CI),
                    np.abs(mdl_F1 - mdl_F1_CI),
                    np.abs(mdl_ROC_AUC - mdl_ROC_AUC_CI),
                    np.abs(mdl_PR_AUC - mdl_PR_AUC_CI),
                    np.abs(mdl_PPV - mdl_PPV_CI),
                    np.abs(mdl_NPV - mdl_NPV_CI),
                ),
                axis=0,
            )

            scores_df = pd.DataFrame(
                {
                    "Model": mdlDict.keys(),
                    "Accuracy": mdl_Acc,
                    "Sensitivity": mdl_sensitivity,
                    "Specificity": mdl_specificity,
                    "f1": mdl_F1,
                    "roc_auc": mdl_ROC_AUC,
                    "pr_auc": mdl_PR_AUC,
                    "PPV": mdl_PPV,
                    "NPV": mdl_NPV,
                }
            )
        elif op_Training == "BSTst" or op_Training == "BSErr":
            # Here, oversample the minority class before bootstrapping
            # you want to resample from the same original sample space

            # https://wiki.python.org/moin/TimeComplexity
            # 5000 repetitions
            # 85 samples

            # mdl_Scores: Dict[str, Any] = dict.fromkeys(mdlDict.keys(), 0)
            mdl_Acc_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_F1_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_ROC_AUC_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_sensitivity_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_specificity_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_PR_AUC_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_PPV_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdl_NPV_bs = np.zeros((len(mdlDict), num_rep_bs))
            mdlAccCI_bs = np.zeros((len(mdlDict), num_rep_bs))

            ## Actually not needed.
            dof_bs = num_rep_bs - 1

            ## Perform over-sampling on the minority class before bootstrapping
            if op_Oversampling == 1:
                ros = RandomOverSampler(
                    random_state=0, sampling_strategy=0.6, shrinkage=1
                )
                predictors_Arr_SAGES_os, responses_SAGES_os = ros.fit_resample(
                    predictors_Arr_SAGES,
                    responses_SAGES,
                )
            elif op_Oversampling == 2:
                sm = SMOTE(random_state=0, sampling_strategy=0.6)
                predictors_Arr_SAGES_os, responses_SAGES_os = sm.fit_resample(
                    predictors_Arr_SAGES,
                    responses_SAGES,
                )
            else:
                predictors_Arr_SAGES_os = predictors_Arr_SAGES
                responses_SAGES_os = responses_SAGES

            ## Bootstrap iteration.
            ## To make it faster, call fit once
            ## And get _score on fitted models.
            # fit_params = fit_params if fit_params is not None else {}
            # fit_params = _check_fit_params(X, fit_params, train)
            # estimator = estimator.set_params(**clone(parameters, safe=False))
            # estimator.fit(X_train, y_train, **fit_params)
            # _score(estimator, X_test, y_test, scorer, error_score="raise"):
            # test_scores = _score(estimator, X_test, y_test, scorer, error_score)
            # result["test_scores"] = test_scores

            for i_mdl, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
                if op_FeatSelect == 1:
                    pipeline = Pipeline(
                        [
                            ("f_selection", SelectKBest(f_classif, k=num_F)),
                            ("classifier", mdl_instance),
                        ]
                    )
                elif op_FeatSelect == 2:
                    pipeline = Pipeline(
                        [
                            (
                                "f_selection",
                                SelectKBest(mutual_info_classif, k=num_F),
                            ),
                            ("classifier", mdl_instance),
                        ]
                    )
                elif op_FeatSelect == 3:
                    # ridge = RidgeCV(alphas=np.logspace(-6,6,num=13))
                    ridge = RidgeClassifier(alpha=1)
                    pipeline = Pipeline(
                        [
                            (
                                "f_selection",
                                SequentialFeatureSelector(
                                    ridge,
                                    n_features_to_select=num_F,
                                    direction="forward",
                                ),
                            ),
                            ("classifier", mdl_instance),
                        ]
                    )
                elif op_FeatSelect == 4:
                    nsc = _NSC.NearestCentroid(metric="manhattan", shrink_threshold=0.5)
                    pipeline = Pipeline(
                        [
                            (
                                "f_selection",
                                SequentialFeatureSelector(
                                    nsc,
                                    n_features_to_select=num_F,
                                    direction="forward",
                                ),
                            ),
                            ("classifier", mdl_instance),
                        ]
                    )
                else:
                    pipeline = mdl_instance

                # Only call fit once to ensure identical model
                pipeline.fit(predictors_Arr_SAGES_os, responses_SAGES_os)
                mdl_Scores[mdl_name] = pipeline
                for iter_bs in range(num_rep_bs):
                    predictors_DUKE_bs, responses_DUKE_bs = resample(
                        predictors_Arr_DUKE,
                        responses_DUKE,
                        n_samples=predictors_Arr_DUKE.shape[0],
                        stratify=responses_DUKE,
                    )

                    responses_DUKE_bs_pred = pipeline.predict(predictors_DUKE_bs)
                    if mdl_name == "NSC Custom EC" or mdl_name == "NSC MH":
                        responses_DUKE_bs_dec_fnc = pipeline.predict(predictors_DUKE_bs)
                    elif (
                        mdl_name == "GNB"
                        or mdl_name == "GNB Prior"
                        or mdl_name == "Tree"
                    ):
                        responses_DUKE_bs_dec_fnc = pipeline.predict_proba(
                            predictors_DUKE_bs
                        )[:, 1]
                    else:
                        responses_DUKE_bs_dec_fnc = pipeline.decision_function(
                            predictors_DUKE_bs
                        )

                    mdl_Acc_bs[i_mdl, iter_bs] = metrics.accuracy_score(
                        responses_DUKE_bs, responses_DUKE_bs_pred
                    )
                    mdl_F1_bs[i_mdl, iter_bs] = metrics.f1_score(
                        responses_DUKE_bs, responses_DUKE_bs_pred
                    )
                    mdl_ROC_AUC_bs[i_mdl, iter_bs] = metrics.roc_auc_score(
                        responses_DUKE_bs, responses_DUKE_bs_dec_fnc
                    )
                    mdl_PR_AUC_bs[i_mdl, iter_bs] = metrics.average_precision_score(
                        responses_DUKE_bs, responses_DUKE_bs_dec_fnc
                    )
                    mdl_PPV_bs[i_mdl, iter_bs] = metrics.precision_score(
                        responses_DUKE_bs, responses_DUKE_bs_pred
                    )
                    mdl_NPV_bs[i_mdl, iter_bs] = metrics.precision_score(
                        responses_DUKE_bs, responses_DUKE_bs_pred, pos_label=0
                    )
                    mdl_sensitivity_bs[i_mdl, iter_bs] = metrics.recall_score(
                        responses_DUKE_bs, responses_DUKE_bs_pred
                    )
                    mdl_specificity_bs[i_mdl, iter_bs] = metrics.recall_score(
                        responses_DUKE_bs, responses_DUKE_bs_pred, pos_label=0
                    )

                if op_Training == "BSErr":
                    responses_SAGES_pred = pipeline.predict(predictors_Arr_SAGES_os)
                    responses_DUKE_pred = pipeline.predict(predictors_Arr_DUKE)
                    if mdl_name == "NSC Custom EC" or mdl_name == "NSC MH":
                        responses_DUKE_dec_fnc = pipeline.predict(predictors_Arr_DUKE)
                    elif (
                        mdl_name == "GNB"
                        or mdl_name == "GNB Prior"
                        or mdl_name == "Tree"
                    ):
                        responses_DUKE_dec_fnc = pipeline.predict_proba(
                            predictors_Arr_DUKE
                        )[:, 1]
                    else:
                        responses_DUKE_dec_fnc = pipeline.decision_function(
                            predictors_Arr_DUKE
                        )

                    ## Save scores to variables.
                    mdl_Acc[i_mdl] = metrics.accuracy_score(
                        responses_DUKE, responses_DUKE_pred
                    )
                    mdl_ROC_AUC[i_mdl] = metrics.roc_auc_score(
                        responses_DUKE, responses_DUKE_dec_fnc
                    )
                    mdl_PR_AUC[i_mdl] = metrics.average_precision_score(
                        responses_DUKE, responses_DUKE_dec_fnc
                    )
                    mdl_PPV[i_mdl] = metrics.precision_score(
                        responses_DUKE, responses_DUKE_pred
                    )
                    mdl_NPV[i_mdl] = metrics.precision_score(
                        responses_DUKE, responses_DUKE_pred, pos_label=0
                    )
                    mdl_sensitivity[i_mdl] = metrics.recall_score(
                        responses_DUKE, responses_DUKE_pred
                    )
                    mdl_specificity[i_mdl] = metrics.recall_score(
                        responses_DUKE, responses_DUKE_pred, pos_label=0
                    )
                    mdl_F1[i_mdl] = 2 / (
                        mdl_PPV[i_mdl] ** (-1) + mdl_sensitivity[i_mdl] ** (-1)
                    )

                    # training perf
                    mdl_Acc_tr[i_mdl] = metrics.accuracy_score(
                        responses_SAGES, responses_SAGES_pred
                    )
                    mdl_ROC_AUC_tr[i_mdl] = metrics.roc_auc_score(
                        responses_SAGES, responses_SAGES_pred
                    )
                    mdl_PR_AUC_tr[i_mdl] = metrics.average_precision_score(
                        responses_SAGES, responses_SAGES_pred
                    )
                    mdl_PPV_tr[i_mdl] = metrics.precision_score(
                        responses_SAGES, responses_SAGES_pred
                    )
                    mdl_NPV_tr[i_mdl] = metrics.precision_score(
                        responses_SAGES, responses_SAGES_pred, pos_label=0
                    )
                    mdl_sensitivity_tr[i_mdl] = metrics.recall_score(
                        responses_SAGES, responses_SAGES_pred
                    )
                    mdl_specificity_tr[i_mdl] = metrics.recall_score(
                        responses_SAGES, responses_SAGES_pred, pos_label=0
                    )
                    mdl_F1_tr[i_mdl] = 2 / (
                        mdl_PPV_tr[i_mdl] ** (-1) + mdl_sensitivity_tr[i_mdl] ** (-1)
                    )
                else:
                    ## Get the mean of different scores of bootstrapped samples.
                    mdl_Acc = np.mean(mdl_Acc_bs, axis=1)
                    mdl_ROC_AUC = np.mean(mdl_ROC_AUC_bs, axis=1)
                    mdl_PR_AUC = np.mean(mdl_PR_AUC_bs, axis=1)
                    mdl_PPV = np.mean(mdl_PPV_bs, axis=1)
                    mdl_NPV = np.mean(mdl_NPV_bs, axis=1)
                    mdl_sensitivity = np.mean(mdl_sensitivity_bs, axis=1)
                    mdl_specificity = np.mean(mdl_specificity_bs, axis=1)
                    mdl_F1 = 2 / (mdl_PPV ** (-1) + mdl_sensitivity ** (-1))

                    ## Get the median of different scores of bootstrapped samples.
                    mdl_Acc_med = np.median(mdl_Acc_bs, axis=1)
                    mdl_ROC_AUC_med = np.median(mdl_ROC_AUC_bs, axis=1)
                    mdl_PR_AUC_med = np.median(mdl_PR_AUC_bs, axis=1)
                    mdl_PPV_med = np.median(mdl_PPV_bs, axis=1)
                    mdl_NPV_med = np.median(mdl_NPV_bs, axis=1)
                    mdl_sensitivity_med = np.median(mdl_sensitivity_bs, axis=1)
                    mdl_specificity_med = np.median(mdl_specificity_bs, axis=1)
                    mdl_F1_med = 2 / (mdl_PPV_med ** (-1) + mdl_sensitivity_med ** (-1))

                ## Write feature names to feature selection text file.
                if isinstance(f_feat, TextIO):
                    f_feat.write("Feature Selection")
                    for el in pipeline.get_feature_names_out(
                        new_psd_transform_SAGES_DF.columns
                    ):
                        f_feat.write(f"{el}\t")
                    f_feat.write("\n\n")

                ## Write model instances' coefficients to csv file.
                if not op_FeatSelect and hasattr(pipeline, "coef_"):
                    coefFN = os.path.join(figDir, f"coeff_{mdl_name}_{fn_opt}.csv")
                    with open(coefFN, "w") as f_coeff:
                        for col_name in new_psd_transform_SAGES_DF.columns:
                            f_coeff.write(f"{col_name},")
                        f_coeff.write("\n")
                        temp_coeff = pipeline.coef_
                        for i_coeff in range(temp_coeff.shape[1]):
                            f_coeff.write(f"{temp_coeff[0,i_coeff]},")
                        f_coeff.write("\n")

            ## Use empirical quantile to compute the 95% confidence intervals.
            # Since distributions are not normal, may need to change
            # the way we calculate CIs.
            mdl_Acc_CI = np.percentile(mdl_Acc_bs, q=[2.5, 97.5], axis=1)
            mdl_F1_CI = np.percentile(mdl_F1_bs, q=[2.5, 97.5], axis=1)
            mdl_ROC_AUC_CI = np.percentile(mdl_ROC_AUC_bs, q=[2.5, 97.5], axis=1)
            mdl_PR_AUC_CI = np.percentile(mdl_PR_AUC_bs, q=[2.5, 97.5], axis=1)
            mdl_PPV_CI = np.percentile(mdl_PPV_bs, q=[2.5, 97.5], axis=1)
            mdl_NPV_CI = np.percentile(mdl_NPV_bs, q=[2.5, 97.5], axis=1)
            mdl_sensitivity_CI = np.percentile(
                mdl_sensitivity_bs, q=[2.5, 97.5], axis=1
            )
            mdl_specificity_CI = np.percentile(
                mdl_specificity_bs, q=[2.5, 97.5], axis=1
            )

            ## Plot histogram of bootstrapped performances.
            fig, axs = plt.subplots(2, 4, figsize=(16, 9.5))
            axs[0, 0].hist(mdl_Acc_bs[1, :])
            axs[0, 0].set_title("Accuracy")

            axs[0, 1].hist(mdl_sensitivity_bs[1, :])
            axs[0, 1].set_title("Sensitivity")

            axs[0, 2].hist(mdl_specificity_bs[1, :])
            axs[0, 2].set_title("Specificity")

            axs[0, 3].hist(mdl_F1_bs[1, :])
            axs[0, 3].set_title("F1")

            axs[1, 0].hist(mdl_ROC_AUC_bs[1, :])
            axs[1, 0].set_title("AUC ROC")

            axs[1, 1].hist(mdl_PR_AUC_bs[1, :])
            axs[1, 1].set_title("AUC PR Curve")

            axs[1, 2].hist(mdl_PPV_bs[1, :])
            axs[1, 2].set_title("PPV")

            axs[1, 3].hist(mdl_PPV_bs[1, :])
            axs[1, 3].set_title("NPV")

            fig.suptitle("LDA CV. Alpha Band.")
            fig.tight_layout()
            bs_lda_cv_fn = os.path.join(figDir, (f"Bootstrap_Hist_LDA_CV.png"))
            fig.savefig(bs_lda_cv_fn)

            fig, axs = plt.subplots(2, 4, figsize=(16, 9.5))
            axs[0, 0].hist(mdl_Acc_bs[2, :])
            axs[0, 0].set_title("Accuracy")

            axs[0, 1].hist(mdl_sensitivity_bs[2, :])
            axs[0, 1].set_title("Sensitivity")

            axs[0, 2].hist(mdl_specificity_bs[2, :])
            axs[0, 2].set_title("Specificity")

            axs[0, 3].hist(mdl_F1_bs[2, :])
            axs[0, 3].set_title("F1")

            axs[1, 0].hist(mdl_ROC_AUC_bs[2, :])
            axs[1, 0].set_title("AUC ROC")

            axs[1, 1].hist(mdl_PR_AUC_bs[2, :])
            axs[1, 1].set_title("AUC PR Curve")

            axs[1, 2].hist(mdl_PPV_bs[2, :])
            axs[1, 2].set_title("PPV")

            axs[1, 3].hist(mdl_PPV_bs[2, :])
            axs[1, 3].set_title("NPV")

            fig.suptitle("LDA LW. Alpha Band.")
            fig.tight_layout()
            bs_lda_lw_fn = os.path.join(figDir, (f"Bootstrap_Hist_LDA_LW.png"))
            fig.savefig(bs_lda_lw_fn)

            fig, axs = plt.subplots(2, 4, figsize=(16, 9.5))
            axs[0, 0].hist(mdl_Acc_bs[4, :])
            axs[0, 0].set_title("Accuracy")

            axs[0, 1].hist(mdl_sensitivity_bs[4, :])
            axs[0, 1].set_title("Sensitivity")

            axs[0, 2].hist(mdl_specificity_bs[4, :])
            axs[0, 2].set_title("Specificity")

            axs[0, 3].hist(mdl_F1_bs[4, :])
            axs[0, 3].set_title("F1")

            axs[1, 0].hist(mdl_ROC_AUC_bs[4, :])
            axs[1, 0].set_title("AUC ROC")

            axs[1, 1].hist(mdl_PR_AUC_bs[4, :])
            axs[1, 1].set_title("AUC PR Curve")

            axs[1, 2].hist(mdl_PPV_bs[4, :])
            axs[1, 2].set_title("PPV")

            axs[1, 3].hist(mdl_PPV_bs[4, :])
            axs[1, 3].set_title("NPV")

            fig.suptitle("NSC MH. Alpha Band.")
            fig.tight_layout()
            bs_nsc_mh_fn = os.path.join(figDir, (f"Bootstrap_Hist_NSC_MH.png"))
            fig.savefig(bs_nsc_mh_fn)

            err_df = np.stack(
                (
                    np.abs(mdl_Acc - mdl_Acc_CI),
                    np.abs(mdl_sensitivity - mdl_sensitivity_CI),
                    np.abs(mdl_specificity - mdl_specificity_CI),
                    np.abs(mdl_F1 - mdl_F1_CI),
                    np.abs(mdl_ROC_AUC - mdl_ROC_AUC_CI),
                    np.abs(mdl_PR_AUC - mdl_PR_AUC_CI),
                    np.abs(mdl_PPV - mdl_PPV_CI),
                    np.abs(mdl_NPV - mdl_NPV_CI),
                ),
                axis=0,
            )

            scores_df = pd.DataFrame(
                {
                    "Model": mdlDict.keys(),
                    "Accuracy": mdl_Acc,
                    "Sensitivity": mdl_sensitivity,
                    "Specificity": mdl_specificity,
                    "f1": mdl_F1,
                    "roc_auc": mdl_ROC_AUC,
                    "pr_auc": mdl_PR_AUC,
                    "PPV": mdl_PPV,
                    "NPV": mdl_NPV,
                }
            )

            scores_tr_df = pd.DataFrame(
                {
                    "Model": mdlDict.keys(),
                    "Accuracy": mdl_Acc_tr,
                    "Sensitivity": mdl_sensitivity_tr,
                    "Specificity": mdl_specificity_tr,
                    "f1": mdl_F1_tr,
                    "roc_auc": mdl_ROC_AUC_tr,
                    "pr_auc": mdl_PR_AUC_tr,
                    "PPV": mdl_PPV_tr,
                    "NPV": mdl_NPV_tr,
                }
            )

        else:
            dof = predictors_Arr.shape[0] - 2
            for i, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
                print(mdl_name)
                if isinstance(f_feat, TextIO):
                    f_feat.write(f"{mdl_name}\n")
                if op_FeatSelect == 1:
                    pipeline = Pipeline(
                        [
                            ("f_selection", SelectKBest(f_classif, k=num_F)),
                            ("classifier", mdl_instance),
                        ]
                    )
                    mdl_Scores[mdl_name] = cross_validate(
                        pipeline,
                        predictors_Arr,
                        responses,
                        cv=skf,
                        scoring=scoring,
                        error_score="raise",
                        return_estimator=True,
                    )
                elif op_FeatSelect == 2:
                    pipeline = Pipeline(
                        [
                            ("f_selection", SelectKBest(mutual_info_classif, k=num_F)),
                            ("classifier", mdl_instance),
                        ]
                    )
                    mdl_Scores[mdl_name] = cross_validate(
                        pipeline,
                        predictors_Arr,
                        responses,
                        cv=skf,
                        scoring=scoring,
                        error_score="raise",
                        return_estimator=True,
                    )
                elif op_FeatSelect == 3:
                    # ridge = RidgeCV(alphas=np.logspace(-6, 6, num=13))
                    ridge = RidgeClassifier(alpha=1)
                    pipeline = Pipeline(
                        [
                            (
                                "f_selection",
                                SequentialFeatureSelector(
                                    ridge,
                                    n_features_to_select=num_F,
                                    direction="forward",
                                ),
                            ),
                            ("classifier", mdl_instance),
                        ]
                    )
                    mdl_Scores[mdl_name] = cross_validate(
                        pipeline,
                        predictors_Arr,
                        responses,
                        cv=skf,
                        scoring=scoring,
                        error_score="raise",
                        return_estimator=True,
                    )
                elif op_FeatSelect == 4:
                    # ridge = RidgeCV(alphas=np.logspace(-6, 6, num=13))
                    nsc = _NSC.NearestCentroid(metric="manhattan", shrink_threshold=0.5)
                    pipeline = Pipeline(
                        [
                            (
                                "f_selection",
                                SequentialFeatureSelector(
                                    nsc, n_features_to_select=num_F, direction="forward"
                                ),
                            ),
                            ("classifier", mdl_instance),
                        ]
                    )
                    mdl_Scores[mdl_name] = cross_validate(
                        pipeline,
                        predictors_Arr,
                        responses,
                        scoring=scoring,
                        error_score="raise",
                        return_estimator=True,
                    )

                else:
                    mdl_Scores[mdl_name] = cross_validate(
                        mdl_instance,
                        predictors_Arr,
                        responses,
                        cv=skf,
                        scoring=scoring,
                        error_score="raise",
                        return_estimator=True,
                    )

                # unknown mean and std
                temp_mdl = mdl_Scores[mdl_name]
                temp_scores = mdl_Scores[mdl_name]["test_accuracy"]
                mdl_Acc[i] = np.mean(temp_scores)
                mdl_ROC_AUC[i] = np.mean(mdl_Scores[mdl_name]["test_roc_auc"])
                mdl_Acc_CI[i] = (
                    t.ppf(1 - (1 - 0.95) / 2, dof)
                    * np.std(temp_scores)
                    / np.sqrt(np.shape(temp_scores)[0])
                )
                mdl_sensitivity[i] = np.mean(mdl_Scores[mdl_name]["test_sensitivity"])
                mdl_specificity[i] = np.mean(mdl_Scores[mdl_name]["test_specificity"])
                mdl_PR_AUC[i] = np.mean(mdl_Scores[mdl_name]["test_pr_auc"])
                mdl_PPV[i] = np.mean(mdl_Scores[mdl_name]["test_PPV"])
                mdl_NPV[i] = np.mean(mdl_Scores[mdl_name]["test_NPV"])
                mdl_F1[i] = 2 / (mdl_PPV[i] ** (-1) + mdl_sensitivity[i] ** (-1))
                if isinstance(f_feat, TextIO):
                    for i_fold in range(len(temp_mdl["estimator"])):
                        f_feat.write(f"Fold {i_fold}:\n")
                        for el in temp_mdl["estimator"][i_fold][
                            0
                        ].get_feature_names_out(fBand_SAGES_DF.columns):
                            f_feat.write(f"{el}\t")
                        f_feat.write("\n\n")

                if not op_FeatSelect and hasattr(temp_mdl["estimator"][0], "coef_"):
                    coefFN = os.path.join(
                        figDir, f"coeff_{mdl_name}_{fn_opt}_{band_name}.csv"
                    )
                    with open(coefFN, "w") as f_coeff:
                        for col_name in fBand_SAGES_DF.columns:
                            f_coeff.write(f"{col_name},")
                        f_coeff.write("\n")
                        for i_fold in range(len(temp_mdl["estimator"])):
                            temp_coeff = temp_mdl["estimator"][i_fold].coef_
                            for i_coeff in range(temp_coeff.shape[1]):
                                f_coeff.write(f"{temp_coeff[0,i_coeff]},")
                            f_coeff.write("\n")

            err_df = pd.DataFrame(
                {
                    "Accuracy": mdl_Acc_CI,
                    "Sensitivity": np.zeros(len(mdlDict)),
                    "Specificity": np.zeros(len(mdlDict)),
                    "f1": np.zeros(len(mdlDict)),
                    "roc_auc": np.zeros(len(mdlDict)),
                    "pr_auc": np.zeros(len(mdlDict)),
                    "PPV": np.zeros(len(mdlDict)),
                    "NPV": np.zeros(len(mdlDict)),
                }
            )

            scores_df = pd.DataFrame(
                {
                    "Model": mdlDict.keys(),
                    "Accuracy": mdl_Acc,
                    "Sensitivity": mdl_sensitivity,
                    "Specificity": mdl_specificity,
                    "f1": mdl_F1,
                    "roc_auc": mdl_ROC_AUC,
                    "pr_auc": mdl_PR_AUC,
                    "PPV": mdl_PPV,
                    "NPV": mdl_NPV,
                }
            )

    ## For plotting
    mpl.use("QtAgg")

    prob_pos = sum(responses_DUKE) / len(responses_DUKE)
    prob_neg = 1 - prob_pos
    samp_pos = prob_pos
    samp_neg = prob_neg

    chc_sim_inst = get_Theor_Chc(prob_pos, prob_neg, samp_pos, samp_neg)

    chc_sim_list = [
        chc_sim_inst.acc,
        chc_sim_inst.sen,
        chc_sim_inst.spc,
        chc_sim_inst.f1,
        chc_sim_inst.roc_auc,
        chc_sim_inst.pr_auc,
        chc_sim_inst.ppv,
        chc_sim_inst.npv,
    ]

    clr_list_drk = get_Clrs_Md.get_clr_drk()

    clr_list_lght = get_Clrs_Md.get_clr_lght()

    ## Start plotting bar charts of performance metrics.
    # title_str = f"Performance of {band_name}. Training DS: " f"{op_Training} {moca_str}"
    title_str = f"Model Selection: SAGES CV Performance of Alpha Power and MoCA."

    ax = scores_df.iloc[:, 1:].plot(
        kind="bar", yerr=err_df, stacked=False, title=title_str, figsize=(15, 12)
    )

    cntns = ax.containers
    num_cntn = len(cntns)

    # make sure this work with grouped bar chart as well
    i_mtrc = 0
    for i_cntn in range(num_cntn):
        if isinstance(cntns[i_cntn], mpl.container.ErrorbarContainer):
            cntns[i_cntn][2][0].set(color=clr_list_lght[i_mtrc % 8])
            cntns[i_cntn][2][0].set(linewidth=2)
        # type(cntn[1]) is mpl.container.BarContainer
        if isinstance(cntns[i_cntn], mpl.container.BarContainer):
            for i_mdl in range(len(cntns[i_cntn])):
                (this_x, this_y) = cntns[i_cntn][i_mdl].get_xy()
                this_width = cntns[i_cntn][i_mdl].get_width()
                plt.plot(
                    [this_x, this_x + this_width],
                    [chc_sim_list[i_mtrc % 8], chc_sim_list[i_mtrc % 8]],
                    "-",
                    color=clr_list_drk[i_mtrc % 8],
                    linewidth=2,
                )
            i_mtrc += 1

    plt.xticks(
        ticks=range(0, len(mdlDict.keys())),
        labels=mdlDict.keys(),
        rotation=45,
        fontsize=14,
    )
    plt.yticks(fontsize=14)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.ylim((0, 1))
    # plt.show()

    if op_Ensemble:
        perf_mean_fig = os.path.join(
            figDir, (f"Performance_Ensemble_{fn_opt}_{band_name}.png")
        )
    else:
        perf_mean_fig = os.path.join(figDir, (f"Performance_{fn_opt}_{band_name}.png"))

    plt.savefig(perf_mean_fig)

    ## training performance
    title_str = (
        f"Training performance of {band_name}. Training DS: {op_Training} {moca_str}"
    )

    ax = scores_tr_df.iloc[:, 1:].plot(
        kind="bar", stacked=False, title=title_str, figsize=(15, 12)
    )

    cntns = ax.containers
    num_cntn = len(cntns)

    # make sure this work with grouped bar chart as well
    i_mtrc = 0
    for i_cntn in range(num_cntn):
        if isinstance(cntns[i_cntn], mpl.container.ErrorbarContainer):
            cntns[i_cntn][2][0].set(color=clr_list_lght[i_mtrc % 8])
            cntns[i_cntn][2][0].set(linewidth=2)
        # type(cntn[1]) is mpl.container.BarContainer
        if isinstance(cntns[i_cntn], mpl.container.BarContainer):
            for i_mdl in range(len(cntns[i_cntn])):
                (this_x, this_y) = cntns[i_cntn][i_mdl].get_xy()
                this_width = cntns[i_cntn][i_mdl].get_width()
                plt.plot(
                    [this_x, this_x + this_width],
                    [chc_sim_list[i_mtrc % 8], chc_sim_list[i_mtrc % 8]],
                    "-",
                    color=clr_list_drk[i_mtrc % 8],
                    linewidth=2,
                )
            i_mtrc += 1

    plt.xticks(
        ticks=range(0, len(mdlDict.keys())),
        labels=mdlDict.keys(),
        rotation=45,
        fontsize=14,
    )
    plt.yticks(fontsize=14)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.ylim((0, 1))
    # plt.show()

    if op_Ensemble:
        perf_mean_fig = os.path.join(
            figDir, (f"Performance_Training_Ensemble_{fn_opt}_{band_name}.png")
        )
    else:
        perf_mean_fig = os.path.join(
            figDir, (f"Performance_Training_{fn_opt}_{band_name}.png")
        )

    plt.savefig(perf_mean_fig)

    ## Plot only LDA CV Results
    plt.figure()
    title_str = (
        f"Performance of LDA CV ({band_name}). Training DS: "
        f"{op_Training} {moca_str}"
    )
    ax = (
        scores_df.iloc[1, 1:]
        .to_frame()
        .transpose()
        .plot(
            kind="bar",
            yerr=np.expand_dims(err_df[:, :, 1], axis=2),
            stacked=False,
            title=title_str,
            figsize=(15, 12),
        )
    )

    cntns = ax.containers
    num_cntn = len(cntns)

    # make sure this work with grouped bar chart as well
    i_mtrc = 0
    for i_cntn in range(num_cntn):
        if isinstance(cntns[i_cntn], mpl.container.ErrorbarContainer):
            cntns[i_cntn][2][0].set(color=clr_list_lght[i_mtrc % 8])
            cntns[i_cntn][2][0].set(linewidth=2)
        # type(cntn[1]) is mpl.container.BarContainer
        if isinstance(cntns[i_cntn], mpl.container.BarContainer):
            (this_x, this_y) = cntns[i_cntn].patches[0].get_xy()
            this_width = cntns[i_cntn].patches[0].get_width()
            plt.plot(
                [this_x, this_x + this_width],
                [chc_sim_list[i_mtrc % 8], chc_sim_list[i_mtrc % 8]],
                "-",
                color=clr_list_drk[i_mtrc % 8],
                linewidth=2,
            )
            i_mtrc += 1

    plt.xticks(ticks=range(0, 1), labels=["LDA CV"], rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.ylim((0, 1))
    # plt.show()

    ## Add baseline performance for DUKE data set
    # chc_acc = (6/51)**2 + (45/51)**2
    # chc_sen = (6/51)**2 / ((6/51)**2 + (6/51)*(45/51))
    # chc_spc = (45/51)**2 / ((45/51)**2 + (6/51)*(45/51))
    # chc_roc_auc = 0.5
    # chc_f1 = 0.5
    # chc_roc_prc = 6/51
    # chc_prc = (6/51)**2 / ((6/51)**2 + (6/51)*(45/51))

    if op_Ensemble:
        perf_mean_fig = os.path.join(
            figDir, (f"Performance_Ensemble_LDACV_{fn_opt}_{band_name}.png")
        )
    else:
        perf_mean_fig = os.path.join(
            figDir, (f"Performance_LDACV_{fn_opt}_{band_name}.png")
        )

    plt.savefig(perf_mean_fig)

    perf_csv_fn = os.path.join(figDir, f"Performance_csv_{fn_opt}_{band_name}.csv")
    with open(perf_csv_fn, "w") as f_perf:
        f_perf.write("Model Name,")
        for metric_name in scoring.keys():
            f_perf.write(
                f"{metric_name} mean, {metric_name} 2.5th, {metric_name} 97.5th,"
            )
        f_perf.write("\n")
        if op_Training == "BS" or op_Training == "BSTst" or op_Training == "BSErr":
            for i_row, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
                f_perf.write(f"{mdl_name},")
                for i_col, col_name in enumerate(scores_df.iloc[:, 1:].columns):
                    f_perf.write(
                        f"{scores_df.iloc[i_row, i_col+1]},"
                        f"{scores_df.iloc[i_row, i_col+1] - err_df[i_col, 0, i_row]},"
                        f"{scores_df.iloc[i_row, i_col+1] + err_df[i_col, 1, i_row]},"
                    )
                f_perf.write("\n")
        else:
            for i_row, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
                f_perf.write(f"{mdl_name},")
                for i_col, col_name in enumerate(scores_df.iloc[:, 1:].columns):
                    f_perf.write(
                        f"{scores_df.iloc[i_row, i_col+1]},"
                        f"{scores_df.iloc[i_row, i_col+1] - err_df.iloc[i_row, i_col]},"
                        f"{scores_df.iloc[i_row, i_col+1] + err_df.iloc[i_row, i_col]},"
                    )
                f_perf.write("\n")

    # training performance csv
    perf_tr_csv_fn = os.path.join(
        figDir, f"Performance_Training_csv_{fn_opt}_{band_name}.csv"
    )
    with open(perf_tr_csv_fn, "w") as f_perf:
        f_perf.write("Model Name,")
        for metric_name in scoring.keys():
            f_perf.write(f"{metric_name} mean,")
        f_perf.write("\n")
        if op_Training == "BS" or op_Training == "BSTst" or op_Training == "BSErr":
            for i_row, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
                f_perf.write(f"{mdl_name},")
                for i_col, col_name in enumerate(scores_tr_df.iloc[:, 1:].columns):
                    f_perf.write(f"{scores_tr_df.iloc[i_row, i_col+1]},")
                f_perf.write("\n")
        else:
            for i_row, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
                f_perf.write(f"{mdl_name},")
                for i_col, col_name in enumerate(scores_df.iloc[:, 1:].columns):
                    f_perf.write(
                        f"{scores_df.iloc[i_row, i_col+1]},"
                        f"{scores_df.iloc[i_row, i_col+1] - err_df.iloc[i_row, i_col]},"
                        f"{scores_df.iloc[i_row, i_col+1] + err_df.iloc[i_row, i_col]},"
                    )
                f_perf.write("\n")

    ## Plot only LDA LW
    plt.figure()
    title_str = (
        f"Performance of LDA LW ({band_name} Hz). Training DS: "
        f"{op_Training} {moca_str}"
    )
    ax = (
        scores_df.iloc[2, 1:]
        .to_frame()
        .transpose()
        .plot(
            kind="bar",
            yerr=np.expand_dims(err_df[:, :, 2], axis=2),
            stacked=False,
            title=title_str,
            figsize=(15, 12),
        )
    )

    cntns = ax.containers
    num_cntn = len(cntns)

    # make sure this work with grouped bar chart as well
    i_mtrc = 0
    for i_cntn in range(num_cntn):
        if isinstance(cntns[i_cntn], mpl.container.ErrorbarContainer):
            cntns[i_cntn][2][0].set(color=clr_list_lght[i_mtrc % 8])
            cntns[i_cntn][2][0].set(linewidth=2)
        # type(cntn[1]) is mpl.container.BarContainer
        if isinstance(cntns[i_cntn], mpl.container.BarContainer):
            (this_x, this_y) = cntns[i_cntn].patches[0].get_xy()
            this_width = cntns[i_cntn].patches[0].get_width()
            plt.plot(
                [this_x, this_x + this_width],
                [chc_sim_list[i_mtrc % 8], chc_sim_list[i_mtrc % 8]],
                "-",
                color=clr_list_drk[i_mtrc % 8],
                linewidth=2,
            )
            i_mtrc += 1

    plt.xticks(ticks=range(0, 1), labels=["LDA LW"], rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(axis="y")
    plt.tight_layout()
    plt.ylim((0, 1))
    # plt.show()

    if op_Ensemble:
        perf_mean_fig = os.path.join(
            figDir, (f"Performance_Ensemble_LDALW_{fn_opt}.png")
        )
    else:
        perf_mean_fig = os.path.join(figDir, (f"Performance_LDALW_{fn_opt}.png"))

    plt.savefig(perf_mean_fig)

    # Plot ROC curve and PR curve
    if op_Training == "BSErr":
        # LDA CV
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))

        # ax1
        title_str = f"Performance of Alpha Powers + MoCA"
        ax_perf = (
            scores_df.iloc[1, 1:]
            .to_frame()
            .transpose()
            .plot(
                kind="bar",
                yerr=np.expand_dims(err_df[:, :, 1], axis=2),
                stacked=False,
                title=title_str,
                ax=ax1,
            )
        )

        ax1.set_title(title_str, fontsize=14)
        cntns = ax_perf.containers
        num_cntn = len(cntns)

        # make sure this work with grouped bar chart as well
        i_mtrc = 0
        for i_cntn in range(num_cntn):
            if isinstance(cntns[i_cntn], mpl.container.ErrorbarContainer):
                cntns[i_cntn][2][0].set(color=clr_list_lght[i_mtrc % 8])
                cntns[i_cntn][2][0].set(linewidth=2)
            # type(cntn[1]) is mpl.container.BarContainer
            if isinstance(cntns[i_cntn], mpl.container.BarContainer):
                (this_x, this_y) = cntns[i_cntn].patches[0].get_xy()
                this_width = cntns[i_cntn].patches[0].get_width()
                ax1.plot(
                    [this_x, this_x + this_width],
                    [chc_sim_list[i_mtrc % 8], chc_sim_list[i_mtrc % 8]],
                    "-",
                    color=clr_list_drk[i_mtrc % 8],
                    linewidth=2,
                )
                i_mtrc += 1

        ax1.set_xticks(ticks=range(0, 1), labels=["LDA CV"], rotation=45, fontsize=14)
        # ax1.set_yticks(ticks=range(0,1), fontsize=14)
        ax1.grid(visible=True, axis="y")
        ax1.set_ylim((0, 1))

        RocCurveDisplay.from_estimator(
            mdl_Scores["LDA CV"], predictors_Arr_DUKE, responses_DUKE, ax=ax2
        )
        ax2.set_title("ROC Curve", fontsize=14)
        ax2.set_xlabel("False Postive Rate", fontsize=14)
        ax2.set_ylabel("True Postive Rate", fontsize=14)
        ax2.set
        PrecisionRecallDisplay.from_estimator(
            mdl_Scores["LDA CV"], predictors_Arr_DUKE, responses_DUKE, ax=ax3
        )
        ax3.set_title("PR Curve", fontsize=14)
        ax3.set_xlabel("Recall", fontsize=14)
        ax3.set_ylabel("Precision", fontsize=14)
        plt.tight_layout()
        if op_Ensemble:
            perf_mean_fig = os.path.join(
                figDir, (f"CurvesPerf_Ensemble_LDACV_{fn_opt}.png")
            )
        else:
            perf_mean_fig = os.path.join(figDir, (f"CurvesPerf_LDACV_{fn_opt}.png"))
        plt.suptitle("Model Validation on INTUIT/PRIME set", fontsize=18)
        plt.savefig(perf_mean_fig)

        # # LDA CV for debugging
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        #
        # RocCurveDisplay.from_estimator(
        #     mdl_Scores["LDA CV"], predictors_Arr_SAGES, responses_SAGES, ax=ax1
        # )
        # ax1.set_title("ROC Curve SAGES")
        # PrecisionRecallDisplay.from_estimator(
        #     mdl_Scores["LDA CV"], predictors_Arr_SAGES, responses_SAGES, ax=ax2
        # )
        # ax2.set_title("PR Curve SAGES")
        # if op_Ensemble:
        #     perf_mean_fig = os.path.join(
        #         figDir, (f"CurvesPerf_Ensemble_LDACV_SAGES_{fn_opt}.png")
        #     )
        # else:
        #     perf_mean_fig = os.path.join(
        #         figDir, (f"CurvesPerf_LDACV_SAGES_{fn_opt}.png")
        #     )
        # plt.suptitle("Performance Curves for LDA CV")
        # plt.savefig(perf_mean_fig)

        # # LDA CV from predictions for debugging
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
        # responses_DUKE_pred = mdl_Scores["LDA CV"].predict(predictors_Arr_DUKE)
        # # responses_DUKE_pred = mdl_Scores["LDA CV"]["estimator"].decision_function(predictors_Arr_DUKE)
        # RocCurveDisplay.from_predictions(responses_DUKE, responses_DUKE_pred, ax=ax1)
        # ax1.set_title("ROC Curve Prediction")
        # PrecisionRecallDisplay.from_predictions(
        #     responses_DUKE, responses_DUKE_pred, ax=ax2
        # )
        # ax2.set_title("PR Curve Prediction")
        # if op_Ensemble:
        #     perf_mean_fig = os.path.join(
        #         figDir, (f"CurvesPerf_Ensemble_LDACV_Prediction_{fn_opt}.png")
        #     )
        # else:
        #     perf_mean_fig = os.path.join(
        #         figDir, (f"CurvesPerf_LDACV_Prediction_{fn_opt}.png")
        #     )
        # plt.suptitle("Performance Curves for LDA CV")
        # plt.savefig(perf_mean_fig)

        # LDA LW
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))

        # ax1
        title_str = f"Performance of Alpha Powers + MoCA"
        ax_perf = (
            scores_df.iloc[2, 1:]
            .to_frame()
            .transpose()
            .plot(
                kind="bar",
                yerr=np.expand_dims(err_df[:, :, 2], axis=2),
                stacked=False,
                title=title_str,
                ax=ax1,
            )
        )

        ax1.set_title(title_str, fontsize=14)
        cntns = ax_perf.containers
        num_cntn = len(cntns)

        # make sure this work with grouped bar chart as well
        i_mtrc = 0
        for i_cntn in range(num_cntn):
            if isinstance(cntns[i_cntn], mpl.container.ErrorbarContainer):
                cntns[i_cntn][2][0].set(color=clr_list_lght[i_mtrc % 8])
                cntns[i_cntn][2][0].set(linewidth=2)
            # type(cntn[1]) is mpl.container.BarContainer
            if isinstance(cntns[i_cntn], mpl.container.BarContainer):
                (this_x, this_y) = cntns[i_cntn].patches[0].get_xy()
                this_width = cntns[i_cntn].patches[0].get_width()
                ax1.plot(
                    [this_x, this_x + this_width],
                    [chc_sim_list[i_mtrc % 8], chc_sim_list[i_mtrc % 8]],
                    "-",
                    color=clr_list_drk[i_mtrc % 8],
                    linewidth=2,
                )
                i_mtrc += 1

        ax1.set_xticks(ticks=range(0, 1), labels=["LDA LW"], rotation=45, fontsize=14)
        # ax1.set_yticks(ticks=range(0,1), fontsize=14)
        ax1.grid(visible=True, axis="y")
        ax1.set_ylim((0, 1))

        RocCurveDisplay.from_estimator(
            mdl_Scores["LDA LW"], predictors_Arr_DUKE, responses_DUKE, ax=ax2
        )
        ax2.set_title("ROC Curve", fontsize=14)
        ax2.set_xlabel("False Postive Rate", fontsize=14)
        ax2.set_ylabel("True Postive Rate", fontsize=14)
        PrecisionRecallDisplay.from_estimator(
            mdl_Scores["LDA LW"], predictors_Arr_DUKE, responses_DUKE, ax=ax3
        )
        ax3.set_title("PR Curve", fontsize=14)
        ax3.set_xlabel("Recall", fontsize=14)
        ax3.set_ylabel("Precision (PPV)", fontsize=14)
        plt.tight_layout()
        if op_Ensemble:
            perf_mean_fig = os.path.join(
                figDir, (f"CurvesPerf_Ensemble_LDALW_{fn_opt}.png")
            )
        else:
            perf_mean_fig = os.path.join(figDir, (f"CurvesPerf_LDALW_{fn_opt}.png"))
        plt.suptitle("Model Validation on INTUIT/PRIME set", fontsize=18)
        plt.savefig(perf_mean_fig)

# plot 2d scatter plots showing corrected vs incorrected predictions
# responses_DUKE_pred
# predictors_Arr_DUKE
# actual responses
# responses_DUKE

responses_DUKE_pred = mdl_Scores["LDA CV"].predict(predictors_Arr_DUKE)

# filter by correct and incorrect responses and by CAM and control
idx_TP = np.logical_and(responses_DUKE_pred == 1, np.array(responses_DUKE.array) == 1)
idx_TN = np.logical_and(responses_DUKE_pred == 0, np.array(responses_DUKE.array) == 0)
idx_FP = np.logical_and(responses_DUKE_pred == 1, np.array(responses_DUKE.array) == 0)
idx_FN = np.logical_and(responses_DUKE_pred == 0, np.array(responses_DUKE.array) == 1)

# fBand_DUKE_DF
if ch_name == "EOEC":
    ch_name_plt_1 = "O2"
    ch_name_plt_2 = "POz"
    ch_name_plt_3 = "O1"

    band_name = "alpha"
    id_ch_1_EO = f"{ch_name_plt_1}_{band_name}_EO"
    id_ch_1_EC = f"{ch_name_plt_1}_{band_name}_EC"

    id_ch_2_EO = f"{ch_name_plt_2}_{band_name}_EO"
    id_ch_2_EC = f"{ch_name_plt_2}_{band_name}_EC"

    id_ch_3_EO = f"{ch_name_plt_3}_{band_name}_EO"
    id_ch_3_EC = f"{ch_name_plt_3}_{band_name}_EC"

    # ch 1 vs ch2
    fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    axs[0].set_title(f"EO")

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_TP, id_ch_1_EO],
        fBand_DUKE_DF.loc[idx_TP, id_ch_2_EO],
        marker="o",
        color="b",
        linestyle="None",
        label="True Positive",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_TN, id_ch_1_EO],
        fBand_DUKE_DF.loc[idx_TN, id_ch_2_EO],
        marker="o",
        color="m",
        linestyle="None",
        label="True Negative",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_FP, id_ch_1_EO],
        fBand_DUKE_DF.loc[idx_FP, id_ch_2_EO],
        marker="x",
        color="m",
        linestyle="None",
        label="False Positive",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_FN, id_ch_1_EO],
        fBand_DUKE_DF.loc[idx_FN, id_ch_2_EO],
        marker="x",
        color="b",
        linestyle="None",
        label="False Negative",
    )

    axs[0].set_xlabel(f"{id_ch_1_EO}", fontsize=12)
    axs[0].set_ylabel(f"{id_ch_2_EO}", fontsize=12)
    axs[0].legend()

    # EC
    axs[1].set_title(f"EC")

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_TP, id_ch_1_EC],
        fBand_DUKE_DF.loc[idx_TP, id_ch_2_EC],
        marker="o",
        color="b",
        linestyle="None",
        label="True Positive",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_TN, id_ch_1_EC],
        fBand_DUKE_DF.loc[idx_TN, id_ch_2_EC],
        marker="o",
        color="m",
        linestyle="None",
        label="True Negative",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_FP, id_ch_1_EC],
        fBand_DUKE_DF.loc[idx_FP, id_ch_2_EC],
        marker="x",
        color="m",
        linestyle="None",
        label="False Positive",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_FN, id_ch_1_EC],
        fBand_DUKE_DF.loc[idx_FN, id_ch_2_EC],
        marker="x",
        color="b",
        linestyle="None",
        label="False Negative",
    )

    axs[1].set_xlabel(f"{id_ch_1_EC}", fontsize=12)
    axs[1].set_ylabel(f"{id_ch_2_EC}", fontsize=12)
    axs[1].legend()

    plt.tight_layout()
    fig.suptitle(rf"Duke Data Set {ch_name_plt_1} vs {ch_name_plt_2}")
    plt.savefig(
        os.path.join(
            figDir,
            f"ScatterPlots_Duke_{band_name}_{ch_name_plt_1}_vs_{ch_name_plt_2}.png",
        )
    )

    # ch1 vs ch3
    fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    axs[0].set_title(f"EO")

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_TP, id_ch_1_EO],
        fBand_DUKE_DF.loc[idx_TP, id_ch_3_EO],
        marker="o",
        color="b",
        linestyle="None",
        label="True Positive",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_TN, id_ch_1_EO],
        fBand_DUKE_DF.loc[idx_TN, id_ch_3_EO],
        marker="o",
        color="m",
        linestyle="None",
        label="True Negative",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_FP, id_ch_1_EO],
        fBand_DUKE_DF.loc[idx_FP, id_ch_3_EO],
        marker="x",
        color="m",
        linestyle="None",
        label="False Positive",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_FN, id_ch_1_EO],
        fBand_DUKE_DF.loc[idx_FN, id_ch_3_EO],
        marker="x",
        color="b",
        linestyle="None",
        label="False Negative",
    )

    axs[0].set_xlabel(f"{id_ch_1_EO}", fontsize=12)
    axs[0].set_ylabel(f"{id_ch_3_EO}", fontsize=12)
    axs[0].legend()

    # EC
    axs[1].set_title(f"EC")

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_TP, id_ch_1_EC],
        fBand_DUKE_DF.loc[idx_TP, id_ch_3_EC],
        marker="o",
        color="b",
        linestyle="None",
        label="True Positive",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_TN, id_ch_1_EC],
        fBand_DUKE_DF.loc[idx_TN, id_ch_3_EC],
        marker="o",
        color="m",
        linestyle="None",
        label="True Negative",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_FP, id_ch_1_EC],
        fBand_DUKE_DF.loc[idx_FP, id_ch_3_EC],
        marker="x",
        color="m",
        linestyle="None",
        label="False Positive",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_FN, id_ch_1_EC],
        fBand_DUKE_DF.loc[idx_FN, id_ch_3_EC],
        marker="x",
        color="b",
        linestyle="None",
        label="False Negative",
    )

    axs[1].set_xlabel(f"{id_ch_1_EC}", fontsize=12)
    axs[1].set_ylabel(f"{id_ch_3_EC}", fontsize=12)
    axs[1].legend()

    plt.tight_layout()
    fig.suptitle(rf"Duke Data Set {ch_name_plt_1} vs {ch_name_plt_3}")
    plt.savefig(
        os.path.join(
            figDir,
            f"ScatterPlots_Duke_{band_name}_{ch_name_plt_1}_vs_{ch_name_plt_3}.png",
        )
    )

    # ch2 vs ch3
    fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
    axs[0].set_title(f"EO")

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_TP, id_ch_2_EO],
        fBand_DUKE_DF.loc[idx_TP, id_ch_3_EO],
        marker="o",
        color="b",
        linestyle="None",
        label="True Positive",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_TN, id_ch_2_EO],
        fBand_DUKE_DF.loc[idx_TN, id_ch_3_EO],
        marker="o",
        color="m",
        linestyle="None",
        label="True Negative",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_FP, id_ch_2_EO],
        fBand_DUKE_DF.loc[idx_FP, id_ch_3_EO],
        marker="x",
        color="m",
        linestyle="None",
        label="False Positive",
    )

    axs[0].plot(
        fBand_DUKE_DF.loc[idx_FN, id_ch_2_EO],
        fBand_DUKE_DF.loc[idx_FN, id_ch_3_EO],
        marker="x",
        color="b",
        linestyle="None",
        label="False Negative",
    )

    axs[0].set_xlabel(f"{id_ch_2_EO}", fontsize=12)
    axs[0].set_ylabel(f"{id_ch_3_EO}", fontsize=12)
    axs[0].legend()

    # EC
    axs[1].set_title(f"EC")

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_TP, id_ch_2_EC],
        fBand_DUKE_DF.loc[idx_TP, id_ch_3_EC],
        marker="o",
        color="b",
        linestyle="None",
        label="True Positive",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_TN, id_ch_2_EC],
        fBand_DUKE_DF.loc[idx_TN, id_ch_3_EC],
        marker="o",
        color="m",
        linestyle="None",
        label="True Negative",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_FP, id_ch_2_EC],
        fBand_DUKE_DF.loc[idx_FP, id_ch_3_EC],
        marker="x",
        color="m",
        linestyle="None",
        label="False Positive",
    )

    axs[1].plot(
        fBand_DUKE_DF.loc[idx_FN, id_ch_2_EC],
        fBand_DUKE_DF.loc[idx_FN, id_ch_3_EC],
        marker="x",
        color="b",
        linestyle="None",
        label="False Negative",
    )

    axs[1].set_xlabel(f"{id_ch_2_EC}", fontsize=12)
    axs[1].set_ylabel(f"{id_ch_3_EC}", fontsize=12)
    axs[1].legend()

    plt.tight_layout()
    fig.suptitle(rf"Duke Data Set {ch_name_plt_2} vs {ch_name_plt_3}")
    plt.savefig(
        os.path.join(
            figDir,
            f"ScatterPlots_Duke_{band_name}_{ch_name_plt_2}_vs_{ch_name_plt_3}.png",
        )
    )

    # Legacy. Plot deviation for NSC. Not supported for bootstrapping.
    test_var = 1
    # if op_FeatSelect:
    #     plt.figure(figsize=(12, 9.5))
    #     feat_Hist = dict(zip(fBand_SAGES_DF.columns,
    #    [0]*len(fBand_SAGES_DF.columns)))
    #     for i, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
    #         temp_mdl = mdl_Scores[mdl_name]
    #         for el in temp_mdl['estimator'][0].get_feature_names_out(
    #                    fBand_SAGES_DF.columns):
    #             feat_Hist[el] = feat_Hist[el]+1
    #
    #     feat_Hist_Nonzero = {k: v for k, v in feat_Hist.items() if v >= 30}
    #
    #     plt.bar(range(len(feat_Hist_Nonzero)), list(feat_Hist_Nonzero.values()))
    #     plt.xticks(range(0, len(feat_Hist_Nonzero)), labels=list(
    #            feat_Hist_Nonzero.keys()),
    #    rotation=45)
    #     plt.tight_layout()
    #     #plt.show()
    #     if op_10_20:
    #         plt.savefig(os.path.join(figDir,f'featBarPlot_{op_Reref}_BP_
    #    {band_name}_
    #     {op_Training}_CrossDS_{moca_str}_{manual_str}_
    #    {featSelect_str}_
    #    {num_F}_{mnt_str}_{unit_str}.png'))
    #     else:
    #         plt.savefig(os.path.join(figDir,f'featBarPlot_{op_Reref}_BP_
    #    {band_name}_
    #    {op_Training}_CrossDS_{moca_str}_{manual_str}_
    #   {featSelect_str}_{num_F}_
    #    {unit_str}.png'))
    #
    #
    # # plot and identify non-zero shrunken centroids_ for each class
    #    in NSC classifier
    # # how to interpret magnitude of non-zero shrunken centroids?
    # # need to show the number of non-zero centroids in each fold
    #
    # # bar chart #1, frequency of features selected across all folds
    # # bar chart #2, number of features selected in each fold
    # # NSC.centroids_ represent equation 18.7, not 18.5 of ESL!!!
    # if not op_FeatSelect:
    #     if op_Training == 'Combined':
    #         plt.figure(figsize=(12, 9.5))
    #         #feat_NSC = dict(zip(new_psd_EO_SAGES_DF.columns, [0]*
    #    len(new_psd_EO_SAGES_DF.columns)))
    #         feat_NSC = np.zeros((2,len(fBand_SAGES_DF.columns)))
    #         temp_mdl = mdl_Scores["NSC Custom EC"]
    #         num_Feat_NSC = np.zeros((len(temp_mdl['estimator']),1))
    #         for i_fold in range(len(temp_mdl['estimator'])):
    #             num_Feat_NSC[i_fold,0] = np.count_nonzero(
    #    np.ravel(temp_mdl['estimator'][i_fold].best_estimator_.deviation))
    #             # temp01[i_fold].best_estimator_.centroids_
    #             for i_class, class_dev in enumerate(
    #    temp_mdl['estimator'][i_fold].best_estimator_.deviation):
    #                 for i_feat, feat_dev in enumerate(class_dev):
    #                     if feat_dev > 0:
    #                         feat_NSC[i_class,i_feat] = feat_NSC[
    #    i_class,i_feat]+1
    #
    #         #feat_NSC_Nonzero = {k: v for k, v in feat_NSC.
    #    items() if v >= 0}
    #         feat_Total_NSC = np.sum(feat_NSC,axis=0)
    #
    #         plt.figure(figsize=(12, 9.5))
    #         plt.bar(range(len(feat_Total_NSC)), list(feat_Total_NSC))
    #         plt.xticks(range(0, len(feat_Total_NSC)), labels=list(
    #    fBand_SAGES_DF.columns), rotation=45)
    #         plt.tight_layout()
    #         #plt.show()
    #         plt.savefig(os.path.join(figDir,
    #    f'featBarPlot_NSC_{op_Reref}_BP_{band_name}_
    #    {op_Training}_CrossDS_{moca_str}_
    #    {manual_str}_{unit_str}.png'))
    #     else:
    #         plt.figure(figsize=(12, 9.5))
    #         #feat_NSC = dict(zip(new_psd_EO_SAGES_DF.columns, [0]*
    #    len(new_psd_EO_SAGES_DF.columns)))
    #         feat_NSC = np.zeros((2,len(fBand_SAGES_DF.columns)))
    #         temp_mdl = mdl_Scores["NSC Custom EC"]
    #         num_Feat_NSC = np.count_nonzero(np.ravel(temp_mdl
    #    ['estimator'].best_estimator_.deviation))
    #         # temp01[i_fold].best_estimator_.centroids_
    #         for i_class, class_dev in enumerate(
    #    temp_mdl['estimator'].best_estimator_.deviation):
    #             for i_feat, feat_dev in enumerate(class_dev):
    #                 if feat_dev > 0:
    #                     feat_NSC[i_class,i_feat] =
    #    feat_NSC[i_class,i_feat]+1
    #
    #         #feat_NSC_Nonzero = {k: v for k, v in
    #    feat_NSC.items() if v >= 0}
    #         feat_Total_NSC = np.sum(feat_NSC,axis=0)
    #
    #         plt.figure(figsize=(12, 9.5))
    #         plt.bar(range(len(feat_Total_NSC)), list(feat_Total_NSC))
    #         plt.xticks(range(0, len(feat_Total_NSC)), labels=list(
    #    fBand_SAGES_DF.columns), rotation=45)
    #         plt.tight_layout()
    #         #plt.show()
    #         plt.savefig(os.path.join(figDir,f'featBarPlot_NSC_
    #    {op_Reref}_BP_{band_name}_{op_Training}_CrossDS_
    #    {moca_str}_{manual_str}_{unit_str}.png'))
    #
    #         # plt.figure(figsize=(12, 9.5))
    #         # plt.bar(range(num_Feat_NSC.shape[0]), np.squeeze(num_Feat_NSC))
    #         # #plt.xticks(range(0, len(feat_NSC_Nonzero)), labels=list
    #    (feat_NSC_Nonzero.keys()), rotation=45)
    #         # plt.xlabel("Fold #")
    #         # plt.ylabel("# of Non-Zero Shrunken Centroids")
    #         # plt.title("# of Non-Zero Shrunken Centroids")
    #         # #plt.ylim((10,25))
    #         # plt.tight_layout()
    #         # #plt.show()
    #         # if op_10_20:
    #         #     if binWidth == 2 and not bin2_startF:
    #         #         plt.savefig(os.path.join(figDir,
    #    f'featBarPlot_NSC_NumFeats_{op_Reref}_BP_{binWidth}EvenHz_All_
    #    Combined_CrossDS_{moca_str}_{manual_str}_
    #    {freq_max}Hz_{unit_str}.png'))
    #         #     else:
    #         #         plt.savefig(os.path.join(figDir,f'featBarPlot_
    #    NSC_NumFeats_{op_Reref}_BP_{binWidth}Hz_All_Combined_CrossDS_
    #    {moca_str}_{manual_str}_{freq_max}Hz_{unit_str}.png'))
    #         # else:
    #         #     if binWidth == 2 and not bin2_startF:
    #         #         plt.savefig(os.path.join(figDir,f'featBarPlot_NSC_
    #    NumFeats_{op_Reref}_BP_{binWidth}EvenHz_All_Combined_CrossDS_
    #    {moca_str}_{manual_str}_{freq_max}Hz_{unit_str}
    #    .png'))
    #         #     else:
    #         #         plt.savefig(os.path.join(figDir,f'featBarPlot_NSC
    #    _NumFeats_{op_Reref}_BP_{binWidth}Hz_All_Combined_CrossDS_
    #    {moca_str}_{manual_str}_{freq_max}Hz_
    #    {unit_str}.png'))
