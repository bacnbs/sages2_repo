"""
STATUS:
  WIP.

FOR PUBLICATION:
  Unknown.

SYNTAX:
  Script.

DESCRIPTION:
  Performs classification of powers of different frequency band powers and plot
  different performance metrics. Additional debugging information are also
  saved. Cross-Validation on SAGES data set. Used for model selection.

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

OPTIONS:
  `op_Scl_CogAssmnt` = {0, 1, 2}.
    Only applies to cognitive assessments, not EEG features.
    -0 : z-score
    -1 : none
    -2 : Distance to the log10 of the median of cognitive assessments.
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
  'freq_min' = positive integer.
    Lower bound of frequency range of PSDs to be used for classification.
  'freq_max' = positive integer.
    Upper bound of frequency range of PSDs to be used for classification
  'binWidth' = positive integer.
    Bandwidth when computing powers. Computed the average, not sum.
  'bin2_startF' = {0, 1}
    Starting frequency when computing band powers.

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
  Created on Jan 23, 2024
  4/4/2024: Add documentation.

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
from contextlib import nullcontext
from typing import Any
from typing import ContextManager
from typing import Dict
from typing import TextIO
from typing import Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from EEGFeaturesExtraction.coreSrc import rem_Multicollinearity
from EEGFeaturesExtraction.coreSrc.chc_lvl_simulation_Md import get_Theor_Chc
from EEGFeaturesExtraction.coreSrc.estimators import _NSC
from EEGFeaturesExtraction.coreSrc.estimators.f1_custom_Md import f1_custom
from EEGFeaturesExtraction.coreSrc.features import get1020Md
from EEGFeaturesExtraction.coreSrc.features import getAgeMd
from EEGFeaturesExtraction.coreSrc.features import getDeliriumStatusMd
from EEGFeaturesExtraction.coreSrc.features import getGCPMd
from EEGFeaturesExtraction.coreSrc.features import getMoCAMD
from EEGFeaturesExtraction.coreSrc.features import getSexMd
from EEGFeaturesExtraction.coreSrc.getModelFnMd import getEnsembleModelFn
from EEGFeaturesExtraction.coreSrc.getModelFnMd import getModelCVFn
from EEGFeaturesExtraction.utils import get_Clrs_Md
from scipy.stats import t
from scipy.stats import zscore
from sklearn import metrics
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model._ridge import RidgeClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


plt.set_loglevel(level="warning")

## Option variables here
op_Local = 0
testCase = 0
dist_Medn = 2
ch_name = "EOEC"
op_Scl_CogAssmnt = 0
op_CogAssmnt = 1
op_sub = 0
op_ManualFeat = 1
op_Ensemble = 0
op_Reref = "car"
op_LowMoCA = 0
op_crr = 1
op_FeatSelect = 0
op_RemCollinearity = 0
op_10_20 = 1
op_mV = 1
op_ROI = 0
num_F = 10
op_Sex = 0
op_Age = 0
relativeFOp = True

freq_min = 6
freq_max = 12
binWidth = 2
bin2_startF = 0

## Option Strings Here
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

if binWidth == 2 and not bin2_startF:
    bin_str = f"{binWidth}EvenHz"
else:
    bin_str = f"{binWidth}Hz"

fn_opt = f"{op_Reref}_{moca_str}_{manual_str}_{ch_name}" f"_{bin_str}_{mnt_str}"

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

if binWidth == 2 and not bin2_startF:
    feat_EO_FN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_BP_{binWidth}EvenHz_EO_{relative_str}_car_mV.csv"
    feat_EC_FN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_BP_{binWidth}EvenHz_EC_{relative_str}_car_mV.csv"
else:
    feat_EO_FN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_BP_{binWidth}Hz_EO_{relative_str}_car_mV.csv"
    feat_EC_FN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_BP_{binWidth}Hz_EC_{relative_str}_car_mV.csv"

feats_EO_DF = pd.read_csv(
    feat_EO_FN,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)
feats_EC_DF = pd.read_csv(
    feat_EC_FN,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)
# feats_bipolar_DF = pd.read_csv(featFN_bipolar, dtype={'# SbjID': pd.StringDtype(), 'TrialNum': np.int32, 'SetFile': pd.StringDtype()})

if op_CogAssmnt == 1:
    cogScore_dict = getMoCAMD.getMoCA(1, op_crr)
elif op_CogAssmnt == 2:
    cogScore_dict = getGCPMd.getGCP(1)
elif op_CogAssmnt == 3:
    if op_sub == 1:
        cogScore_dict = getMoCAMD.getSpatialEx()
    elif op_sub == 2:
        cogScore_dict = getMoCAMD.getAttention()
    elif op_sub == 3:
        cogScore_dict = getMoCAMD.getMemory()
    elif op_sub == 4:
        moCA_dict = getMoCAMD.getMoCA(1, op_crr)
        spa_dict = getMoCAMD.getSpatialEx()
        att_dict = getMoCAMD.getAttention()
        mem_dict = getMoCAMD.getMemory()
        cogScore_dict = {}
        for k in moCA_dict.keys():
            cogScore_dict[k] = spa_dict[k] + att_dict[k] + mem_dict[k]
elif op_CogAssmnt == 4:
    temp_dict = getMoCAMD.getMoCA(1, op_crr)
    temp_dict_round = {k: round(v) for (k, v) in temp_dict.items()}
    cogScore_dict = getMoCAMD.convert_MoCA_MMSE(temp_dict_round)

sex_dict = getSexMd.getSex(1)
age_dict = getAgeMd.getAge(1)

for key, value in sex_dict.items():
    sex_dict[key] = 1 if value == "M" else 0

# Compute SPR first before normalizing
# psd_df = feats_EO_DF.filter(regex=f"(# SbjID)|(TrialNum)|(^{ch_name}_)", axis="columns")
new_psd_EO_df = feats_EO_DF.drop("SetFile", axis="columns", inplace=False)
new_psd_EC_df = feats_EC_DF.drop("SetFile", axis="columns", inplace=False)

del feats_EO_DF
del feats_EC_DF

new_psd_EO_df.set_index(["# SbjID", "TrialNum"], inplace=True)
new_psd_EC_df.set_index(["# SbjID", "TrialNum"], inplace=True)

freqs = list(zip(*(col.split("_") for col in new_psd_EO_df)))[-1]
freqs_flt = [float(f) for f in freqs]
freqs_flt_arr = np.asarray(freqs_flt)
cols_freqs = np.where(
    np.logical_and(freqs_flt_arr >= freq_min, freqs_flt_arr <= freq_max)
)
# ind_min = (np.abs(freqs_flt_arr - freq_min)).argmin()
# ind_max = (np.abs(freqs_flt_arr - freq_max)).argmin()
new_psd_EO_df = new_psd_EO_df.iloc[:, cols_freqs[0]]

freqs = list(zip(*(col.split("_") for col in new_psd_EC_df)))[-1]
freqs_flt = [float(f) for f in freqs]
freqs_flt_arr = np.asarray(freqs_flt)
cols_freqs = np.where(
    np.logical_and(freqs_flt_arr >= freq_min, freqs_flt_arr <= freq_max)
)
# ind_min = (np.abs(freqs_flt_arr - freq_min)).argmin()
# ind_max = (np.abs(freqs_flt_arr - freq_max)).argmin()
new_psd_EC_df = new_psd_EC_df.iloc[:, cols_freqs[0]]

new_psd_df = new_psd_EO_df.join(
    new_psd_EC_df, on=["# SbjID", "TrialNum"], how="inner", lsuffix="_EO", rsuffix="_EC"
)

## For debugging
new_psd_EO_df.to_csv(path_or_buf=os.path.join(csvTestDir, "new_psd_EO_SAGES_DF.csv"))

new_psd_EC_df.to_csv(path_or_buf=os.path.join(csvTestDir, "new_psd_EC_SAGES_DF.csv"))

new_psd_df.to_csv(path_or_buf=os.path.join(csvTestDir, "new_psd_SAGES_DF.csv"))

# rename columns before performing operations
# numpy.ndarray
# alpha_col_names_orig = psd_df.columns.values
#
# freqs = list(zip(*(col.split('_') for col in new_psd_EO_df)))[-1]
# freqs_flt = [float(f) for f in freqs]
# freqs_flt_arr = np.asarray(freqs_flt)
# cols_freqs = np.where(np.logical_and(freqs_flt_arr>=freq_min, freqs_flt_arr<=freq_max))
# # ind_min = (np.abs(freqs_flt_arr - freq_min)).argmin()
# # ind_max = (np.abs(freqs_flt_arr - freq_max)).argmin()
# new_psd_EO_df = new_psd_EO_df.iloc[:,cols_freqs[0]]
# n_freq = len(freqs)
#
# # SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
# # meanAlpha_df._is_copy = weakref to Dataframe (True)
# # meanAlpha_df._is_view = False
# # Ignore or not? Do we want it to be a view or a copy?
# # I want it to be a copy but why same variable name?
# mean_alpha_copy_df = psd_df.copy()
#
# mean_alpha_copy_df.rename(
#     columns=dict(
#         zip(
#             alpha_col_names_orig.tolist(),
#             chn_names)
#         ),
#     inplace=True)

# compute ratio first before z-score


# if op_ROI:
#     roi_dict = getROIMd.getROICh(op_10_20)
#     for i_col, (col_name, col_list) in enumerate(roi_dict.items()):
#         spr_df[col_name] = spr_df[col_list].mean(axis=1)
#     spr_df = spr_df[roi_dict.keys()]

# spr_df.to_csv(val_FN, header=True, sep=',')

# Normalize all samples

if dist_Medn == 1:
    # distance to the median
    new_psd_transform_df = new_psd_df.apply(lambda x: np.abs((x - x.median())))
elif dist_Medn == 2:
    # distance between log10 spr and the log10 of the median
    new_psd_transform_df = new_psd_df.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - np.log10(x.median())) != 0,
            np.abs(np.log10(x) - np.log10(x.median())),
            np.NaN,
        )
    )
elif dist_Medn == 3:
    whole_medn = np.log10(np.median(new_psd_df))
    new_psd_transform_df = new_psd_df.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - whole_medn) != 0,
            np.abs(np.log10(x) - whole_medn),
            np.NaN,
        )
    )
elif dist_Medn == 4:
    scaler = MinMaxScaler(feature_range=(-1, 1))

    feats_EO_DF = scaler.fit_transform(feats_EO_DF)

    feats_EO_DF = feats_EO_DF.apply(
        lambda x: np.where(
            np.abs(np.arcsinh(x, where=(x != 0)) - np.arcsinh(x.median())) != 0,
            np.abs(np.arcsinh(x, where=(x != 0)) - np.arcsinh(x.median())),
            0,
        )
    )
elif dist_Medn == 6:
    new_psd_transform_df = new_psd_df.apply(lambda x: 10 * np.log10(x))
else:
    new_psd_transform_df = new_psd_df.apply(scipy.stats.zscore)

new_psd_transform_df.replace(np.NaN, 0, inplace=True)

if op_RemCollinearity:
    new_psd_transform_df = rem_Multicollinearity.remove_vif(
        new_psd_transform_df, thresh=40
    )

if op_CogAssmnt:
    new_psd_transform_df["MoCA"] = new_psd_transform_df.index.get_level_values(0).map(
        cogScore_dict
    )
    if op_Scl_CogAssmnt == 0:
        new_psd_transform_df["MoCA"] = (
            new_psd_transform_df["MoCA"].to_frame().apply(zscore, nan_policy="omit")
        )
    elif op_Scl_CogAssmnt == 2:
        new_psd_transform_df["MoCA"] = (
            new_psd_transform_df["MoCA"]
            .to_frame()
            .apply(
                lambda x: np.where(
                    np.abs(np.log10(x) - np.log10(x.median())) != 0,
                    np.abs(np.log10(x) - np.log10(x.median())),
                    np.NaN,
                )
            )
        )
    elif op_Scl_CogAssmnt == 4:
        pass

## Add sex to dataframes.
if op_Sex:
    new_psd_transform_df["Sex"] = new_psd_transform_df.index.get_level_values(0).map(
        sex_dict
    )
    # new_psd_transform_df['Sex'] = new_psd_transform_df['Sex'].to_frame().apply(zscore, nan_policy='omit')

## Add age to dataframes.
if op_Age:
    new_psd_transform_df["Age"] = new_psd_transform_df.index.get_level_values(0).map(
        age_dict
    )
    new_psd_transform_df["Age"] = (
        new_psd_transform_df["Age"].to_frame().apply(zscore, nan_policy="omit")
    )
    # qt = QuantileTransformer(output_distribution='normal')
    # plt.subplot(2, 2, 1)
    # plt.hist(new_psd_transform_df['Age'], bins='auto')
    # new_psd_transform_df['Age'] = qt.fit_transform(new_psd_transform_df['Age'].to_numpy().reshape(-1,1))
    # plt.subplot(2, 2, 2)
    # plt.hist(new_psd_transform_df['Age'], bins='auto')
    # plt.show()

if op_LowMoCA:
    LowMoCA_List = [k for k, v in cogScore_dict.items() if v < 21]

    new_psd_transform_df = new_psd_transform_df.loc[
        new_psd_transform_df.index.get_level_values(0).isin(LowMoCA_List)
    ]

CAM = getDeliriumStatusMd.getDeliriumStatus(1)

responses = new_psd_transform_df.index.get_level_values(0).to_frame()
# try {0,1} and {-1,1} encoding. will get different results.
responses["Y"] = np.where(responses["# SbjID"].isin(CAM), 1, 0)
responses = responses["Y"]

# Does NOT support group as advertised!!!
# It's actually a scaling coefficient alpha that weight
# between L1 and L2 norm.
# https://github.com/scikit-learn-contrib/lightning

chns_10_20 = get1020Md.get1020()
chns_10_20.append("MoCA")

# if op_10_20 and not op_ROI:
#     new_psd_transform_df = new_psd_transform_df[chns_10_20]

# new_psd_transform_df = new_psd_transform_df.filter(regex=f"(# SbjID)|(TrialNum)|(^{c4_str}_)", axis="columns")

# predictors_Arr = new_psd_transform_df.to_numpy()

if op_Ensemble:
    mdlDict = getEnsembleModelFn()
else:
    mdlDict = getModelCVFn()
    # mdlDict = getModelNSCFn()

dof = new_psd_transform_df.shape[0] - 2

scoring = {
    "accuracy": make_scorer(metrics.accuracy_score),
    "sensitivity": make_scorer(metrics.recall_score),
    "specificity": make_scorer(metrics.recall_score, pos_label=0),
    "f1": make_scorer(metrics.f1_score),
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "PPV": make_scorer(metrics.precision_score),
    "NPV": make_scorer(metrics.precision_score, pos_label=0),
}

# stratified cross-validation
# avoid hyperparameter, use cross-validation
# skf = StratifiedKFold(n_splits=5, shuffle=True)
if op_LowMoCA:
    skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=10)
else:
    skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10)

if op_FeatSelect:
    feat_Sel_FN = os.path.join(figDir, f"featList_BP_EOEC_{fn_opt}.txt")

mdl_Scores: Dict[str, Any] = dict.fromkeys(mdlDict.keys(), 0)
mdl_Acc = np.zeros((len(mdlDict)))
mdl_F1 = np.zeros((len(mdlDict)))
mdl_ROC_AUC = np.zeros((len(mdlDict)))
mdl_PR_AUC = np.zeros((len(mdlDict)))
mdl_sensitivity = np.zeros((len(mdlDict)))
mdl_specificity = np.zeros((len(mdlDict)))
mdl_PR_AUC = np.zeros((len(mdlDict)))
mdl_PPV = np.zeros((len(mdlDict)))
mdl_NPV = np.zeros((len(mdlDict)))

mdl_Acc_CI = np.zeros((len(mdlDict)))
mdl_Sen_CI = np.zeros((len(mdlDict)))
mdl_Spc_CI = np.zeros((len(mdlDict)))
mdl_F1_CI = np.zeros((len(mdlDict)))
mdl_ROC_AUC_CI = np.zeros((len(mdlDict)))
mdl_PR_AUC_CI = np.zeros((len(mdlDict)))
mdl_PPV_CI = np.zeros((len(mdlDict)))
mdl_NPV_CI = np.zeros((len(mdlDict)))

if op_10_20:
    if binWidth == 2 and not bin2_startF:
        manual_feat_selection = [
            "O1_10_EO",
            "O2_10_EO",
            "POz_10_EO",
            "O1_8_EC",
            "O2_8_EC",
            "POz_8_EC",
        ]
        # manual_feat_selection = [
        #     "O1_10_EO",
        #     "O2_10_EO",
        #     "POz_10_EO",
        #     "O1_8_EC",
        #     "O2_8_EC",
        #     "POz_8_EC",
        #     "Fp1_8_EC",
        #     "Fp2_8_EC",
        #     "Fp1_8_EO",
        #     "Fp2_8_EO",
        # ]

    else:
        # Best one: "EO9_EC8"
        manual_feat_selection = [
            "O1_11_EO",
            "O2_11_EO",
            "POz_11_EO",
            "O1_9_EC",
            "O2_9_EC",
            "POz_9_EC",
        ]
        # manual_feat_selection = [
        #     "O1_9_EO",
        #     "O2_9_EO",
        #     "POz_9_EO",
        #     "O1_9_EC",
        #     "O2_9_EC",
        #     "POz_9_EC",
        # ]
        # manual_feat_selection = [
        #     "O1_8_EC",
        #     "O2_8_EC",
        #     "POz_8_EC",
        #     "O1_9_EC",
        #     "O2_9_EC",
        #     "POz_9_EC",
        # ]
        # manual_feat_selection = [
        #     "O1_9_EC",
        #     "O2_9_EC",
        #     "POz_9_EC",
        # ]
else:
    if binWidth == 2 and not bin2_startF:
        manual_feat_selection = [
            "O1_10_EO",
            "O2_10_EO",
            "Oz_10_EO",
            "POz_10_EO",
            "PO3_10_EO",
            "PO4_10_EO",
            "O1_8_EC",
            "O2_8_EC",
            "Oz_8_EC",
            "POz_8_EC",
            "PO3_8_EC",
            "PO4_8_EC",
        ]
    else:
        manual_feat_selection = [
            "O1_11_EO",
            "O2_11_EO",
            "Oz_11_EO",
            "POz_11_EO",
            "PO3_11_EO",
            "PO4_11_EO",
            "O1_9_EC",
            "O2_9_EC",
            "Oz_9_EC",
            "POz_9_EC",
            "PO3_9_EC",
            "PO4_9_EC",
        ]

if op_CogAssmnt:
    manual_feat_selection.append("MoCA")
if op_Sex:
    manual_feat_selection.append("Sex")
if op_Age:
    manual_feat_selection.append("Age")

if op_ManualFeat:
    new_psd_transform_df = new_psd_transform_df[manual_feat_selection]

if op_10_20 and not op_ManualFeat and not op_ROI:
    new_psd_transform_df = new_psd_transform_df[chns_10_20]

predictors_Arr = new_psd_transform_df.to_numpy()

# For debugging
np.savetxt(
    os.path.join(csvTestDir, "predictors_Arr.csv"), predictors_Arr, delimiter=","
)
np.savetxt(os.path.join(csvTestDir, "responses.csv"), responses, delimiter=",")

f_feat: Union[TextIO, ContextManager[None]] = (
    open(feat_Sel_FN, "w") if op_FeatSelect else nullcontext()
)

# if f_feat is None:
#     raise RuntimeError(f"Failed to open {feat_Sel_FN}")
# assert isinstance(f_feat, TextIO)

with f_feat:
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
        elif op_FeatSelect == 2:
            pipeline = Pipeline(
                [
                    ("f_selection", SelectKBest(mutual_info_classif, k=num_F)),
                    ("classifier", mdl_instance),
                ]
            )
        elif op_FeatSelect == 3:
            # ridge = RidgeCV(alphas=np.logspace(-6, 6, num=13))
            ridge = RidgeClassifier(alpha=1)
            pipeline = Pipeline(
                [
                    (
                        "f_selection",
                        SequentialFeatureSelector(
                            ridge, n_features_to_select=num_F, direction="forward"
                        ),
                    ),
                    ("classifier", mdl_instance),
                ]
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

        else:
            pipeline = mdl_instance

        mdl_Scores[mdl_name] = cross_validate(
            pipeline,
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
        mdl_PR_AUC[i] = np.mean(mdl_Scores[mdl_name]["test_pr_auc"])
        mdl_sensitivity[i] = np.mean(mdl_Scores[mdl_name]["test_sensitivity"])
        mdl_specificity[i] = np.mean(mdl_Scores[mdl_name]["test_specificity"])
        mdl_PR_AUC[i] = np.mean(mdl_Scores[mdl_name]["test_pr_auc"])
        mdl_PPV[i] = np.mean(mdl_Scores[mdl_name]["test_PPV"])
        mdl_NPV[i] = np.mean(mdl_Scores[mdl_name]["test_NPV"])
        mdl_F1[i] = 2 / (mdl_PPV[i] ** (-1) + mdl_sensitivity[i] ** (-1))

        ## Compute CI
        mdl_Acc_CI[i] = (
            t.ppf(1 - (1 - 0.95) / 2, dof)
            * np.std(temp_scores)
            / np.sqrt(np.shape(temp_scores)[0])
        )

        mdl_Sen_CI[i] = (
            t.ppf(1 - (1 - 0.95) / 2, dof)
            * np.std(mdl_Scores[mdl_name]["test_sensitivity"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_sensitivity"])[0])
        )

        mdl_Spc_CI[i] = (
            t.ppf(1 - (1 - 0.95) / 2, dof)
            * np.std(mdl_Scores[mdl_name]["test_specificity"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_specificity"])[0])
        )

        mdl_F1_CI[i] = (
            t.ppf(1 - (1 - 0.95) / 2, dof)
            * np.std(mdl_Scores[mdl_name]["test_f1"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_f1"])[0])
        )

        mdl_ROC_AUC_CI[i] = (
            t.ppf(1 - (1 - 0.95) / 2, dof)
            * np.std(mdl_Scores[mdl_name]["test_roc_auc"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_roc_auc"])[0])
        )

        mdl_PR_AUC_CI[i] = (
            t.ppf(1 - (1 - 0.95) / 2, dof)
            * np.std(mdl_Scores[mdl_name]["test_pr_auc"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_pr_auc"])[0])
        )

        mdl_PPV_CI[i] = (
            t.ppf(1 - (1 - 0.95) / 2, dof)
            * np.std(mdl_Scores[mdl_name]["test_PPV"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_PPV"])[0])
        )

        mdl_NPV_CI[i] = (
            t.ppf(1 - (1 - 0.95) / 2, dof)
            * np.std(mdl_Scores[mdl_name]["test_NPV"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_NPV"])[0])
        )

        if isinstance(f_feat, TextIO):
            for i_fold in range(len(temp_mdl["estimator"])):
                f_feat.write(f"Fold {i_fold}:\n")
                for el in temp_mdl["estimator"][i_fold][0].get_feature_names_out(
                    new_psd_transform_df.columns
                ):
                    f_feat.write(f"{el}\t")
                f_feat.write("\n\n")

        if not op_FeatSelect and hasattr(temp_mdl["estimator"][0], "coef_"):
            coefFN = rf"coeff_{mdl_name}_BP_{fn_opt}.csv"
            coefFN = os.path.join(figDir, coefFN)
            with open(coefFN, "w") as f_coeff:
                for col_name in new_psd_transform_df.columns:
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
        "Sensitivity": mdl_Sen_CI,
        "Specificity": mdl_Spc_CI,
        "f1": mdl_F1_CI,
        "roc_auc": mdl_ROC_AUC_CI,
        "pr_auc": mdl_PR_AUC_CI,
        "PPV": mdl_PPV_CI,
        "NPV": mdl_NPV_CI,
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

mpl.use("QtAgg")

prob_pos = sum(responses) / len(responses)
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

title_str = f"Model Selection: Performance of Sub-Alpha Power + MoCA"

ax = scores_df.iloc[:, 1:].plot(
    kind="bar", yerr=err_df, stacked=False, title=title_str, figsize=(15, 12)
)

ax.legend(loc="lower right")
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

if op_Ensemble:
    perf_mean_fig = os.path.join(figDir, (f"Performance_Ensemble_{fn_opt}.png"))
else:
    perf_mean_fig = os.path.join(figDir, (f"Performance_{fn_opt}.png"))

plt.savefig(perf_mean_fig)

perf_csv_fn = os.path.join(figDir, f"Performance_csv_{fn_opt}.csv")
with open(perf_csv_fn, "w") as f_perf:
    f_perf.write("Model Name,")
    for metric_name in scoring.keys():
        f_perf.write(f"{metric_name} mean, {metric_name} 2.5th, {metric_name} 97.5th,")
    f_perf.write("\n")
    for i_row, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
        f_perf.write(f"{mdl_name},")
        for i_col, col_name in enumerate(scores_df.iloc[:, 1:].columns):
            f_perf.write(
                f"{scores_df.iloc[i_row, i_col+1]},"
                f"{scores_df.iloc[i_row, i_col+1] - err_df.iloc[i_row, i_col]},"
                f"{scores_df.iloc[i_row, i_col+1] + err_df.iloc[i_row, i_col]},"
            )
        f_perf.write("\n")

# save to csv
if op_FeatSelect:
    plt.figure(figsize=(12, 9.5))
    feat_Hist = dict(
        zip(new_psd_transform_df.columns, [0] * len(new_psd_transform_df.columns))
    )
    for i, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
        temp_mdl = mdl_Scores[mdl_name]
        for i_fold in range(len(temp_mdl["estimator"])):
            for el in temp_mdl["estimator"][i_fold][0].get_feature_names_out(
                new_psd_transform_df.columns
            ):
                feat_Hist[el] = feat_Hist[el] + 1

    feat_Hist_Nonzero = {k: v for k, v in feat_Hist.items() if v >= 30}

    plt.bar(range(len(feat_Hist_Nonzero)), list(feat_Hist_Nonzero.values()))
    plt.xticks(
        range(0, len(feat_Hist_Nonzero)),
        labels=list(feat_Hist_Nonzero.keys()),
        rotation=45,
    )
    plt.tight_layout()

    feat_Hist_Fig = os.path.join(figDir, f"featBarPlot_{fn_opt}.png")

    plt.savefig(feat_Hist_Fig)


# plot and identify non-zero shrunken centroids_ for each class in NSC classifier
# how to interpret magnitude of non-zero shrunken centroids?
# need to show the number of non-zero centroids in each fold

# bar chart #1, frequency of features selected across all folds
# bar chart #2, number of features selected in each fold
# NSC.centroids_ represent equation 18.7, not 18.5 of ESL!!!
if not op_FeatSelect and not op_Ensemble:
    plt.figure(figsize=(12, 9.5))
    # feat_NSC = dict(zip(new_psd_EO_df.columns, [0]*len(new_psd_EO_df.columns)))
    feat_NSC = np.zeros((2, len(new_psd_transform_df.columns)))
    temp_mdl = mdl_Scores["NSC Custom EC"]
    num_Feat_NSC = np.zeros((len(temp_mdl["estimator"]), 1))
    for i_fold in range(len(temp_mdl["estimator"])):
        num_Feat_NSC[i_fold, 0] = np.count_nonzero(
            np.ravel(temp_mdl["estimator"][i_fold].best_estimator_.deviations_)
        )
        # temp01[i_fold].best_estimator_.centroids_
        for i_class, class_dev in enumerate(
            temp_mdl["estimator"][i_fold].best_estimator_.deviations_
        ):
            for i_feat, feat_dev in enumerate(class_dev):
                if feat_dev > 0:
                    feat_NSC[i_class, i_feat] = feat_NSC[i_class, i_feat] + 1

    # feat_NSC_Nonzero = {k: v for k, v in feat_NSC.items() if v >= 0}
    feat_Total_NSC = np.sum(feat_NSC, axis=0)

    plt.figure(figsize=(12, 9.5))
    plt.bar(range(len(feat_Total_NSC)), list(feat_Total_NSC))
    plt.xticks(
        range(0, len(feat_Total_NSC)),
        labels=list(new_psd_transform_df.columns),
        rotation=45,
    )
    plt.tight_layout()
    # plt.show()
    feat_Hist_NSC_Fig = os.path.join(figDir, f"featBarPlot_NSC_{fn_opt}.png")
    plt.savefig(feat_Hist_NSC_Fig)

    plt.figure(figsize=(12, 9.5))
    plt.bar(range(num_Feat_NSC.shape[0]), np.squeeze(num_Feat_NSC))
    # plt.xticks(range(0, len(feat_NSC_Nonzero)), labels=list(feat_NSC_Nonzero.keys()), rotation=45)
    plt.xlabel("Fold #")
    plt.ylabel("# of Non-Zero Shrunken Centroids")
    plt.title("# of Non-Zero Shrunken Centroids")
    # plt.ylim((10,25))
    plt.tight_layout()

    feat_Hist_NSC_NumFeats_Fig = os.path.join(
        figDir, f"featBarPlot_NSC_NumFeats_{fn_opt}.png"
    )
    plt.savefig(feat_Hist_NSC_NumFeats_Fig)
