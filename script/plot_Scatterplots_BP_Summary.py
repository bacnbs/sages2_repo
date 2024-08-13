"""
Created on Jan 26, 2024

Plot scatterplot of bandpower for each Hz

Summary Figure for Publication
Pick one channel and use it as a representative

@author: mning
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from EEGFeaturesExtraction.coreSrc.features import getDeliriumStatusMd


plt.set_loglevel(level="warning")
testCase = 0

ch_name = "Fp1"
binWidth = 2
bin2_startF = 0
bin_Freq = "10"
is_EO = 1
# dist_Medn only applies to SPR and not cognitive assessment scores
dist_Medn = 2
# transformation options for cognitive assessment scores
# op_Scl_CogAssmnt:
#        0 for z-score
#        1 for none
#        2 for DistMdnLog10
op_Scl_CogAssmnt = 0
opCogAssmnt = 0
opMax = 0
op_Reref = "car"
op_crr = 1
op_10_20 = 0
# for debugging purpose
op_debug = 0
# c4_str = 'C4'
# very slow!!!
if opMax == 1:
    max_str = "Max"
elif opMax == 2:
    max_str = "MaxMean"
else:
    max_str = "Mean"
if opCogAssmnt == 1:
    if op_crr:
        moca_str = "MoCACorrected"
    else:
        moca_str = "MoCA"
elif opCogAssmnt == 2:
    moca_str = "GCP"
else:
    moca_str = "NoMoCA"
if is_EO:
    eo_str = "EO"
else:
    eo_str = "EC"
# relativeFOp is for getSpecStatsMd.bandpower
relativeFOp = True
if dist_Medn == 1:
    scale_str = "DistMdn"
elif dist_Medn == 2:
    scale_str = "DistMdnLog10"
elif dist_Medn == 3:
    scale_str = "DistMdnWholeLog10"
elif dist_Medn == 0:
    scale_str = "ZTrnsfrm"
elif dist_Medn == 6:
    scale_str = "10Log10"
if relativeFOp:
    relative_str = "Relative"
else:
    relative_str = "Absolute"

fig_Dir_SAGES = (
    r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data"
    r"\Baseline_rsEEG_preprocessing\DATA_visualization\Group\PSDs"
)

fig_Dir_Duke = (
    r"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
    r"\Duke Preop EEG Data\DATA\DATA_visualization\Group\PSDs"
)

if binWidth == 2 and not bin2_startF:
    feat_SAGES_EO_FN = (
        rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
        rf"\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj"
        rf"_BP_{binWidth}EvenHz_EO_{relative_str}_car_mV.csv"
    )

    feat_SAGES_EC_FN = (
        rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
        rf"\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj"
        rf"_BP_{binWidth}EvenHz_EC_{relative_str}_car_mV.csv"
    )

    feat_DUKE_EO_FN = (
        rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
        rf"\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj"
        rf"_BP_{binWidth}EvenHz_EO_{relative_str}_car_mV.csv"
    )

    feat_DUKE_EC_FN = (
        rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
        rf"\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj"
        rf"_BP_{binWidth}EvenHz_EC_{relative_str}_car_mV.csv"
    )
else:
    feat_SAGES_EO_FN = (
        rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
        rf"\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj"
        rf"_BP_{binWidth}Hz_EO_{relative_str}_car_mV.csv"
    )

    feat_SAGES_EC_FN = (
        rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
        rf"\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj"
        rf"_BP_{binWidth}Hz_EC_{relative_str}_car_mV.csv"
    )

    feat_DUKE_EO_FN = (
        rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
        rf"\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj"
        rf"_BP_{binWidth}Hz_EO_{relative_str}_car_mV.csv"
    )

    feat_DUKE_EC_FN = (
        rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
        rf"\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj"
        rf"_BP_{binWidth}Hz_EC_{relative_str}_car_mV.csv"
    )

feats_SAGES_EO_DF = pd.read_csv(
    feat_SAGES_EO_FN,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)

feats_SAGES_EC_DF = pd.read_csv(
    feat_SAGES_EC_FN,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)

feats_DUKE_EO_DF = pd.read_csv(
    feat_DUKE_EO_FN,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)

feats_DUKE_EC_DF = pd.read_csv(
    feat_DUKE_EC_FN,
    dtype={
        "# SbjID": pd.StringDtype(),
        "TrialNum": np.int32,
        "SetFile": pd.StringDtype(),
    },
)

## For debugging
csvTestDir = (
    r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data"
    "\Baseline_rsEEG_preprocessing\csv_tests"
)

new_psd_SAGES_EO_DF = feats_SAGES_EO_DF.drop("SetFile", axis="columns", inplace=False)
new_psd_SAGES_EC_DF = feats_SAGES_EC_DF.drop("SetFile", axis="columns", inplace=False)
new_psd_DUKE_EO_DF = feats_DUKE_EO_DF.drop("SetFile", axis="columns", inplace=False)
new_psd_DUKE_EC_DF = feats_DUKE_EC_DF.drop("SetFile", axis="columns", inplace=False)

new_psd_SAGES_EO_DF.set_index(["# SbjID", "TrialNum"], inplace=True)
new_psd_SAGES_EC_DF.set_index(["# SbjID", "TrialNum"], inplace=True)
new_psd_DUKE_EO_DF.set_index(["# SbjID", "TrialNum"], inplace=True)
new_psd_DUKE_EC_DF.set_index(["# SbjID", "TrialNum"], inplace=True)

new_psd_SAGES_df = new_psd_SAGES_EO_DF.join(
    new_psd_SAGES_EC_DF,
    on=["# SbjID", "TrialNum"],
    how="inner",
    lsuffix="_EO",
    rsuffix="_EC",
)

new_psd_DUKE_df = new_psd_DUKE_EO_DF.join(
    new_psd_DUKE_EC_DF,
    on=["# SbjID", "TrialNum"],
    how="inner",
    lsuffix="_EO",
    rsuffix="_EC",
)

# debug
if op_debug:
    new_psd_SAGES_EO_DF.to_csv(
        path_or_buf=os.path.join(csvTestDir, "new_psd_EO_SAGES_DF.csv")
    )

    new_psd_SAGES_EC_DF.to_csv(
        path_or_buf=os.path.join(csvTestDir, "new_psd_EC_SAGES_DF.csv")
    )

    new_psd_SAGES_df.to_csv(
        path_or_buf=os.path.join(csvTestDir, "new_psd_SAGES_DF.csv")
    )

# delete to clear up memory
del new_psd_SAGES_EO_DF, new_psd_SAGES_EC_DF, new_psd_DUKE_EO_DF, new_psd_DUKE_EC_DF

if dist_Medn == 1:
    # distance to the median
    new_psd_SAGES_transform_df = new_psd_SAGES_df.apply(
        lambda x: np.abs((x - x.median()))
    )
    new_psd_DUKE_transform_df = new_psd_DUKE_df.apply(
        lambda x: np.abs((x - x.median()))
    )
elif dist_Medn == 2:
    # distance between log10 spr and the log10 of the median
    new_psd_SAGES_transform_df = new_psd_SAGES_df.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - np.log10(x.median())) != 0,
            np.abs(np.log10(x) - np.log10(x.median())),
            np.NaN,
        )
    )

    sages_Median = np.log10(new_psd_SAGES_df.median(axis=0))

    new_psd_DUKE_transform_df = new_psd_DUKE_df.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - sages_Median.loc[x.name]) != 0,
            np.abs(np.log10(x) - sages_Median.loc[x.name]),
            np.NaN,
        ),
        axis=0,
    )
    new_psd_SAGES_transform_df.replace(np.NaN, 0, inplace=True)
    new_psd_DUKE_transform_df.replace(np.NaN, 0, inplace=True)
elif dist_Medn == 3:
    whole_medn = np.log10(np.median(new_psd_SAGES_df))
    new_psd_SAGES_transform_df = new_psd_SAGES_df.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - whole_medn) != 0,
            np.abs(np.log10(x) - whole_medn),
            np.NaN,
        )
    )
    whole_medn = np.log10(np.median(new_psd_DUKE_df))
    new_psd_DUKE_transform_df = new_psd_DUKE_df.apply(
        lambda x: np.where(
            np.abs(np.log10(x) - whole_medn) != 0,
            np.abs(np.log10(x) - whole_medn),
            np.NaN,
        )
    )
    new_psd_SAGES_transform_df.replace(np.NaN, 0, inplace=True)
    new_psd_DUKE_transform_df.replace(np.NaN, 0, inplace=True)
elif dist_Medn == 0:
    new_psd_SAGES_transform_df = new_psd_SAGES_df.apply(scipy.stats.zscore)
    new_psd_DUKE_transform_df = new_psd_DUKE_df.apply(scipy.stats.zscore)
elif dist_Medn == 6:
    new_psd_SAGES_transform_df = new_psd_SAGES_df.apply(lambda x: 10 * np.log10(x))
    new_psd_DUKE_transform_df = new_psd_DUKE_df.apply(lambda x: 10 * np.log10(x))

new_psd_SAGES_df = new_psd_SAGES_df.apply(lambda x: 10 * np.log10(x))
new_psd_DUKE_df = new_psd_DUKE_df.apply(lambda x: 10 * np.log10(x))

CAM_SAGES = getDeliriumStatusMd.getDeliriumStatus(1)
CAM_DUKE = getDeliriumStatusMd.getDeliriumStatus(0)

# Untransformed
feats_DF_CAM_SAGES_Untrnsfrm_Sbj = new_psd_SAGES_df[
    new_psd_SAGES_df.index.isin(CAM_SAGES, level=0)
]
feats_DF_Ctrl_SAGES_Untrnsfrm_Sbj = new_psd_SAGES_df[
    ~new_psd_SAGES_df.index.isin(CAM_SAGES, level=0)
]

feats_DF_CAM_SAGES_Untrnsfrm_Grp = (
    feats_DF_CAM_SAGES_Untrnsfrm_Sbj.mean(axis="rows").to_frame().T
)
feats_DF_Ctrl_SAGES_Untrnsfrm_Grp = (
    feats_DF_Ctrl_SAGES_Untrnsfrm_Sbj.mean(axis="rows").to_frame().T
)

feats_DF_CAM_DUKE_Untrnsfrm_Sbj = new_psd_DUKE_df[
    new_psd_DUKE_df.index.isin(CAM_DUKE, level=0)
]
feats_DF_Ctrl_DUKE_Untrnsfrm_Sbj = new_psd_DUKE_df[
    ~new_psd_DUKE_df.index.isin(CAM_DUKE, level=0)
]

feats_DF_CAM_DUKE_Untrnsfrm_Grp = (
    feats_DF_CAM_DUKE_Untrnsfrm_Sbj.mean(axis="rows").to_frame().T
)
feats_DF_Ctrl_DUKE_Untrnsfrm_Grp = (
    feats_DF_Ctrl_DUKE_Untrnsfrm_Sbj.mean(axis="rows").to_frame().T
)

# Transformed
feats_DF_CAM_SAGES_Trnsfrm_Sbj = new_psd_SAGES_transform_df[
    new_psd_SAGES_transform_df.index.isin(CAM_SAGES, level=0)
]
feats_DF_Ctrl_SAGES_Trnsfrm_Sbj = new_psd_SAGES_transform_df[
    ~new_psd_SAGES_transform_df.index.isin(CAM_SAGES, level=0)
]

feats_DF_CAM_SAGES_Trnsfrm_Grp = (
    feats_DF_CAM_SAGES_Trnsfrm_Sbj.mean(axis="rows").to_frame().T
)
feats_DF_Ctrl_SAGES_Trnsfrm_Grp = (
    feats_DF_Ctrl_SAGES_Trnsfrm_Sbj.mean(axis="rows").to_frame().T
)

feats_DF_CAM_DUKE_Trnsfrm_Sbj = new_psd_DUKE_transform_df[
    new_psd_DUKE_transform_df.index.isin(CAM_DUKE, level=0)
]
feats_DF_Ctrl_DUKE_Trnsfrm_Sbj = new_psd_DUKE_transform_df[
    ~new_psd_DUKE_transform_df.index.isin(CAM_DUKE, level=0)
]

feats_DF_CAM_DUKE_Trnsfrm_Grp = (
    feats_DF_CAM_DUKE_Trnsfrm_Sbj.mean(axis="rows").to_frame().T
)
feats_DF_Ctrl_DUKE_Trnsfrm_Grp = (
    feats_DF_Ctrl_DUKE_Trnsfrm_Sbj.mean(axis="rows").to_frame().T
)

id_Chn_bin_Freq_EO = f"{ch_name}_{bin_Freq}_EO"
id_Chn_bin_Freq_EC = f"{ch_name}_{bin_Freq}_EC"

ttst_SAGES_Untrnsfrm_EO = scipy.stats.ttest_ind(
    feats_DF_CAM_SAGES_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    feats_DF_Ctrl_SAGES_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    equal_var=False,
)
ttst_SAGES_Untrnsfrm_EC = scipy.stats.ttest_ind(
    feats_DF_CAM_SAGES_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    feats_DF_Ctrl_SAGES_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    equal_var=False,
)
ttst_DUKE_Untrnsfrm_EO = scipy.stats.ttest_ind(
    feats_DF_CAM_DUKE_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    feats_DF_Ctrl_DUKE_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    equal_var=False,
)
ttst_DUKE_Untrnsfrm_EC = scipy.stats.ttest_ind(
    feats_DF_CAM_DUKE_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    feats_DF_Ctrl_DUKE_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    equal_var=False,
)

ttst_SAGES_Trnsfrm_EO = scipy.stats.ttest_ind(
    feats_DF_CAM_SAGES_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    feats_DF_Ctrl_SAGES_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    equal_var=False,
)
ttst_SAGES_Trnsfrm_EC = scipy.stats.ttest_ind(
    feats_DF_CAM_SAGES_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    feats_DF_Ctrl_SAGES_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    equal_var=False,
)
ttst_DUKE_Trnsfrm_EO = scipy.stats.ttest_ind(
    feats_DF_CAM_DUKE_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    feats_DF_Ctrl_DUKE_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    equal_var=False,
)
ttst_DUKE_Trnsfrm_EC = scipy.stats.ttest_ind(
    feats_DF_CAM_DUKE_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    feats_DF_Ctrl_DUKE_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    equal_var=False,
)

pval_list_Untrnsfrm = [
    ttst_SAGES_Untrnsfrm_EO.pvalue,
    ttst_SAGES_Untrnsfrm_EC.pvalue,
    ttst_DUKE_Untrnsfrm_EO.pvalue,
    ttst_DUKE_Untrnsfrm_EC.pvalue,
]

pval_list_Trnsfrm = [
    ttst_SAGES_Trnsfrm_EO.pvalue,
    ttst_SAGES_Trnsfrm_EC.pvalue,
    ttst_DUKE_Trnsfrm_EO.pvalue,
    ttst_DUKE_Trnsfrm_EC.pvalue,
]

x_ticks_4zip = ["SAGES EO", "SAGES EC", "DUKE EO", "DUKE EC"]

x_ticks_lbls_Untrnsfrm = [
    x_name + f" ({pval:.2f})"
    for (x_name, pval) in zip(x_ticks_4zip, pval_list_Untrnsfrm)
]
x_ticks_lbls_Trnsfrm = [
    x_name + f" ({pval:.2f})" for (x_name, pval) in zip(x_ticks_4zip, pval_list_Trnsfrm)
]

x_tick_idcs = [0, 1, 2, 3]

fig, axs = plt.subplots(figsize=(10, 6), nrows=1, ncols=2)
axs[0].set_title("Untransformed")

axs[0].plot(
    -0.2 * np.ones(feats_DF_Ctrl_SAGES_Untrnsfrm_Sbj.shape[0]),
    feats_DF_Ctrl_SAGES_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    marker="o",
    color="b",
    linestyle="None",
)

axs[0].plot(
    0.2 * np.ones(feats_DF_CAM_SAGES_Untrnsfrm_Sbj.shape[0]),
    feats_DF_CAM_SAGES_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    marker="o",
    color="m",
    linestyle="None",
)

axs[0].plot(
    -0.2,
    feats_DF_Ctrl_SAGES_Untrnsfrm_Grp.loc[0, id_Chn_bin_Freq_EO],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[0].plot(
    0.2,
    feats_DF_CAM_SAGES_Untrnsfrm_Grp.loc[0, id_Chn_bin_Freq_EO],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)

axs[0].plot(
    0.8 * np.ones(feats_DF_Ctrl_SAGES_Untrnsfrm_Sbj.shape[0]),
    feats_DF_Ctrl_SAGES_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    marker="o",
    color="b",
    linestyle="None",
)

axs[0].plot(
    1.2 * np.ones(feats_DF_CAM_SAGES_Untrnsfrm_Sbj.shape[0]),
    feats_DF_CAM_SAGES_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    marker="o",
    color="m",
    linestyle="None",
)

axs[0].plot(
    0.8,
    feats_DF_Ctrl_SAGES_Untrnsfrm_Grp.loc[0, id_Chn_bin_Freq_EC],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[0].plot(
    1.2,
    feats_DF_CAM_SAGES_Untrnsfrm_Grp.loc[0, id_Chn_bin_Freq_EC],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)

# DUKE
axs[0].plot(
    1.8 * np.ones(feats_DF_Ctrl_DUKE_Untrnsfrm_Sbj.shape[0]),
    feats_DF_Ctrl_DUKE_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    marker="o",
    color="b",
    linestyle="None",
)

axs[0].plot(
    2.2 * np.ones(feats_DF_CAM_DUKE_Untrnsfrm_Sbj.shape[0]),
    feats_DF_CAM_DUKE_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    marker="o",
    color="m",
    linestyle="None",
)

axs[0].plot(
    1.8,
    feats_DF_Ctrl_DUKE_Untrnsfrm_Grp.loc[0, id_Chn_bin_Freq_EO],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[0].plot(
    2.2,
    feats_DF_CAM_DUKE_Untrnsfrm_Grp.loc[0, id_Chn_bin_Freq_EO],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)

axs[0].plot(
    2.8 * np.ones(feats_DF_Ctrl_DUKE_Untrnsfrm_Sbj.shape[0]),
    feats_DF_Ctrl_DUKE_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    marker="o",
    color="b",
    linestyle="None",
)

axs[0].plot(
    3.2 * np.ones(feats_DF_CAM_DUKE_Untrnsfrm_Sbj.shape[0]),
    feats_DF_CAM_DUKE_Untrnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    marker="o",
    color="m",
    linestyle="None",
)

axs[0].plot(
    2.8,
    feats_DF_Ctrl_DUKE_Untrnsfrm_Grp.loc[0, id_Chn_bin_Freq_EC],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[0].plot(
    3.2,
    feats_DF_CAM_DUKE_Untrnsfrm_Grp.loc[0, id_Chn_bin_Freq_EC],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[0].set_xticks(x_tick_idcs, x_ticks_lbls_Untrnsfrm, rotation=45)
axs[0].set_ylabel("Log Power [$10*log_{10}(\mu V^2/Hz)$]", fontsize=12)
# plt.ylim(0,2)
plt.tight_layout()

# Transformed
axs[1].set_title("Transformed")
axs[1].plot(
    -0.2 * np.ones(feats_DF_Ctrl_SAGES_Trnsfrm_Sbj.shape[0]),
    feats_DF_Ctrl_SAGES_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    marker="o",
    color="b",
    linestyle="None",
)

axs[1].plot(
    0.2 * np.ones(feats_DF_CAM_SAGES_Trnsfrm_Sbj.shape[0]),
    feats_DF_CAM_SAGES_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    marker="o",
    color="m",
    linestyle="None",
)

axs[1].plot(
    -0.2,
    feats_DF_Ctrl_SAGES_Trnsfrm_Grp.loc[0, id_Chn_bin_Freq_EO],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[1].plot(
    0.2,
    feats_DF_CAM_SAGES_Trnsfrm_Grp.loc[0, id_Chn_bin_Freq_EO],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)

axs[1].plot(
    0.8 * np.ones(feats_DF_Ctrl_SAGES_Trnsfrm_Sbj.shape[0]),
    feats_DF_Ctrl_SAGES_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    marker="o",
    color="b",
    linestyle="None",
)

axs[1].plot(
    1.2 * np.ones(feats_DF_CAM_SAGES_Trnsfrm_Sbj.shape[0]),
    feats_DF_CAM_SAGES_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    marker="o",
    color="m",
    linestyle="None",
)

axs[1].plot(
    0.8,
    feats_DF_Ctrl_SAGES_Trnsfrm_Grp.loc[0, id_Chn_bin_Freq_EC],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[1].plot(
    1.2,
    feats_DF_CAM_SAGES_Trnsfrm_Grp.loc[0, id_Chn_bin_Freq_EC],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)

# DUKE
axs[1].plot(
    1.8 * np.ones(feats_DF_Ctrl_DUKE_Trnsfrm_Sbj.shape[0]),
    feats_DF_Ctrl_DUKE_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    marker="o",
    color="b",
    linestyle="None",
)

axs[1].plot(
    2.2 * np.ones(feats_DF_CAM_DUKE_Trnsfrm_Sbj.shape[0]),
    feats_DF_CAM_DUKE_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EO],
    marker="o",
    color="m",
    linestyle="None",
)

axs[1].plot(
    1.8,
    feats_DF_Ctrl_DUKE_Trnsfrm_Grp.loc[0, id_Chn_bin_Freq_EO],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[1].plot(
    2.2,
    feats_DF_CAM_DUKE_Trnsfrm_Grp.loc[0, id_Chn_bin_Freq_EO],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)

axs[1].plot(
    2.8 * np.ones(feats_DF_Ctrl_DUKE_Trnsfrm_Sbj.shape[0]),
    feats_DF_Ctrl_DUKE_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    marker="o",
    color="b",
    linestyle="None",
)

axs[1].plot(
    3.2 * np.ones(feats_DF_CAM_DUKE_Trnsfrm_Sbj.shape[0]),
    feats_DF_CAM_DUKE_Trnsfrm_Sbj.loc[:, id_Chn_bin_Freq_EC],
    marker="o",
    color="m",
    linestyle="None",
)

axs[1].plot(
    2.8,
    feats_DF_Ctrl_DUKE_Trnsfrm_Grp.loc[0, id_Chn_bin_Freq_EC],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)
axs[1].plot(
    3.2,
    feats_DF_CAM_DUKE_Trnsfrm_Grp.loc[0, id_Chn_bin_Freq_EC],
    marker="s",
    markersize=10,
    color="k",
    linestyle="None",
)

axs[1].set_xticks(x_tick_idcs, x_ticks_lbls_Trnsfrm, rotation=45)
axs[1].set_ylabel("A.U.", fontsize=12)
# plt.ylim(0,2)
plt.tight_layout()
fig.suptitle(rf"{ch_name} {bin_Freq}Hz BP Scatter Plots")
plt.savefig(os.path.join(fig_Dir_SAGES, f"ScatterPlot_{bin_Freq}Hz_BP_{ch_name}.png"))
