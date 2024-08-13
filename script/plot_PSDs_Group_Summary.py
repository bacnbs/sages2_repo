"""
Created on Jan 25, 2024

Plot scatterplot of bandpower for each Hz

Summary Figure for Publication
Pick one channel and use it as a representative

@author: mning
"""
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from EEGFeaturesExtraction.coreSrc.features import getDeliriumStatusMd


plt.set_loglevel(level="warning")
mpl.use("QtAgg")
testCase = 0

ch_name = "POz"
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
freq_min = 1
freq_max = 26
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

feat_SAGES_EO_FN = (
    rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
    rf"\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_"
    rf"PSD_EO_{relative_str}_car_mV.csv"
)

feat_SAGES_EC_FN = (
    rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
    rf"\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_"
    rf"PSD_EC_{relative_str}_car_mV.csv"
)

feat_DUKE_EO_FN = (
    rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
    rf"\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj_PSD_"
    rf"EO_{relative_str}_car_mV.csv"
)

feat_DUKE_EC_FN = (
    rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II"
    rf"\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj_PSD_"
    rf"EC_{relative_str}_car_mV.csv"
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

new_psd_SAGES_EO_DF = new_psd_SAGES_EO_DF.apply(lambda x: 10 * np.log10(x))
new_psd_SAGES_EC_DF = new_psd_SAGES_EC_DF.apply(lambda x: 10 * np.log10(x))
new_psd_DUKE_EO_DF = new_psd_DUKE_EO_DF.apply(lambda x: 10 * np.log10(x))
new_psd_DUKE_EC_DF = new_psd_DUKE_EC_DF.apply(lambda x: 10 * np.log10(x))

CAM_SAGES = getDeliriumStatusMd.getDeliriumStatus(1)
CAM_DUKE = getDeliriumStatusMd.getDeliriumStatus(0)

# Untransformed
feats_DF_CAM_SAGES_EO_Untrnsfrm_Sbj = new_psd_SAGES_EO_DF[
    new_psd_SAGES_EO_DF.index.isin(CAM_SAGES, level=0)
]
feats_DF_Ctrl_SAGES_EO_Untrnsfrm_Sbj = new_psd_SAGES_EO_DF[
    ~new_psd_SAGES_EO_DF.index.isin(CAM_SAGES, level=0)
]

feats_DF_CAM_SAGES_EO_Untrnsfrm_Grp = feats_DF_CAM_SAGES_EO_Untrnsfrm_Sbj.mean(
    axis="rows"
)
feats_DF_Ctrl_SAGES_EO_Untrnsfrm_Grp = feats_DF_Ctrl_SAGES_EO_Untrnsfrm_Sbj.mean(
    axis="rows"
)

feats_DF_CAM_SAGES_EC_Untrnsfrm_Sbj = new_psd_SAGES_EC_DF[
    new_psd_SAGES_EC_DF.index.isin(CAM_SAGES, level=0)
]
feats_DF_Ctrl_SAGES_EC_Untrnsfrm_Sbj = new_psd_SAGES_EC_DF[
    ~new_psd_SAGES_EC_DF.index.isin(CAM_SAGES, level=0)
]

feats_DF_CAM_SAGES_EC_Untrnsfrm_Grp = feats_DF_CAM_SAGES_EC_Untrnsfrm_Sbj.mean(
    axis="rows"
)
feats_DF_Ctrl_SAGES_EC_Untrnsfrm_Grp = feats_DF_Ctrl_SAGES_EC_Untrnsfrm_Sbj.mean(
    axis="rows"
)

feats_DF_CAM_DUKE_EO_Untrnsfrm_Sbj = new_psd_DUKE_EO_DF[
    new_psd_DUKE_EO_DF.index.isin(CAM_DUKE, level=0)
]
feats_DF_Ctrl_DUKE_EO_Untrnsfrm_Sbj = new_psd_DUKE_EO_DF[
    ~new_psd_DUKE_EO_DF.index.isin(CAM_DUKE, level=0)
]

feats_DF_CAM_DUKE_EO_Untrnsfrm_Grp = feats_DF_CAM_DUKE_EO_Untrnsfrm_Sbj.mean(
    axis="rows"
)
feats_DF_Ctrl_DUKE_EO_Untrnsfrm_Grp = feats_DF_Ctrl_DUKE_EO_Untrnsfrm_Sbj.mean(
    axis="rows"
)

feats_DF_CAM_DUKE_EC_Untrnsfrm_Sbj = new_psd_DUKE_EC_DF[
    new_psd_DUKE_EC_DF.index.isin(CAM_DUKE, level=0)
]
feats_DF_Ctrl_DUKE_EC_Untrnsfrm_Sbj = new_psd_DUKE_EC_DF[
    ~new_psd_DUKE_EC_DF.index.isin(CAM_DUKE, level=0)
]

feats_DF_CAM_DUKE_EC_Untrnsfrm_Grp = feats_DF_CAM_DUKE_EC_Untrnsfrm_Sbj.mean(
    axis="rows"
)
feats_DF_Ctrl_DUKE_EC_Untrnsfrm_Grp = feats_DF_Ctrl_DUKE_EC_Untrnsfrm_Sbj.mean(
    axis="rows"
)

sem_SAGES_CAM_EO_PSD_DF = feats_DF_CAM_SAGES_EO_Untrnsfrm_Sbj.sem()
sem_SAGES_Ctrl_EO_PSD_DF = feats_DF_Ctrl_SAGES_EO_Untrnsfrm_Sbj.sem()
sem_DUKE_CAM_EO_PSD_DF = feats_DF_CAM_DUKE_EO_Untrnsfrm_Sbj.sem()
sem_DUKE_Ctrl_EO_PSD_DF = feats_DF_Ctrl_DUKE_EO_Untrnsfrm_Sbj.sem()

sem_SAGES_CAM_EC_PSD_DF = feats_DF_CAM_SAGES_EC_Untrnsfrm_Sbj.sem()
sem_SAGES_Ctrl_EC_PSD_DF = feats_DF_Ctrl_SAGES_EC_Untrnsfrm_Sbj.sem()
sem_DUKE_CAM_EC_PSD_DF = feats_DF_CAM_DUKE_EC_Untrnsfrm_Sbj.sem()
sem_DUKE_Ctrl_EC_PSD_DF = feats_DF_Ctrl_DUKE_EC_Untrnsfrm_Sbj.sem()

# for standard deviation of PSD, use delta-method to estimate Var(log10(PSD))
ch_CAM_SAGES_EO_PSD_DF = feats_DF_CAM_SAGES_EO_Untrnsfrm_Grp.filter(
    regex=f"(^{ch_name}_)"
).T
ch_CAM_SAGES_EO_PSD_SEM_DF = sem_SAGES_CAM_EO_PSD_DF.filter(regex=f"(^{ch_name}_)").T

ch_Ctrl_SAGES_EO_PSD_DF = feats_DF_Ctrl_SAGES_EO_Untrnsfrm_Grp.filter(
    regex=f"(^{ch_name}_)"
).T
ch_Ctrl_SAGES_EO_PSD_SEM_DF = sem_SAGES_Ctrl_EO_PSD_DF.filter(regex=f"(^{ch_name}_)").T

ch_CAM_DUKE_EO_PSD_DF = feats_DF_CAM_DUKE_EO_Untrnsfrm_Grp.filter(
    regex=f"(^{ch_name}_)"
).T
ch_CAM_DUKE_EO_PSD_SEM_DF = sem_DUKE_CAM_EO_PSD_DF.filter(regex=f"(^{ch_name}_)").T

ch_Ctrl_DUKE_EO_PSD_DF = feats_DF_Ctrl_DUKE_EO_Untrnsfrm_Grp.filter(
    regex=f"(^{ch_name}_)"
).T
ch_Ctrl_DUKE_EO_PSD_SEM_DF = sem_DUKE_Ctrl_EO_PSD_DF.filter(regex=f"(^{ch_name}_)").T

ch_CAM_SAGES_EC_PSD_DF = feats_DF_CAM_SAGES_EC_Untrnsfrm_Grp.filter(
    regex=f"(^{ch_name}_)"
).T
ch_CAM_SAGES_EC_PSD_SEM_DF = sem_SAGES_CAM_EC_PSD_DF.filter(regex=f"(^{ch_name}_)").T

ch_Ctrl_SAGES_EC_PSD_DF = feats_DF_Ctrl_SAGES_EC_Untrnsfrm_Grp.filter(
    regex=f"(^{ch_name}_)"
).T
ch_Ctrl_SAGES_EC_PSD_SEM_DF = sem_SAGES_Ctrl_EC_PSD_DF.filter(regex=f"(^{ch_name}_)").T

ch_CAM_DUKE_EC_PSD_DF = feats_DF_CAM_DUKE_EC_Untrnsfrm_Grp.filter(
    regex=f"(^{ch_name}_)"
).T
ch_CAM_DUKE_EC_PSD_SEM_DF = sem_DUKE_CAM_EC_PSD_DF.filter(regex=f"(^{ch_name}_)").T

ch_Ctrl_DUKE_EC_PSD_DF = feats_DF_Ctrl_DUKE_EC_Untrnsfrm_Grp.filter(
    regex=f"(^{ch_name}_)"
).T
ch_Ctrl_DUKE_EC_PSD_SEM_DF = sem_DUKE_Ctrl_EC_PSD_DF.filter(regex=f"(^{ch_name}_)").T

# freq_nm_df = ch_CAM_DUKE_EO_PSD_DF.filter(regex=f"^Cz", axis="columns")
freqs_SAGES = list(zip(*(i.split("_") for (i, row) in ch_CAM_SAGES_EO_PSD_DF.items())))[
    -1
]
freqs_flt = [float(f) for f in freqs_SAGES]
freqs_flt_arr = np.asarray(freqs_flt)
cols_freqs = np.where(
    np.logical_and(freqs_flt_arr >= freq_min, freqs_flt_arr <= freq_max)
)

ch_CAM_SAGES_EO_PSD_DF = ch_CAM_SAGES_EO_PSD_DF.iloc[cols_freqs[0]]
ch_CAM_SAGES_EO_PSD_SEM_DF = ch_CAM_SAGES_EO_PSD_SEM_DF.iloc[cols_freqs[0]]

ch_CAM_SAGES_EC_PSD_DF = ch_CAM_SAGES_EC_PSD_DF.iloc[cols_freqs[0]]
ch_CAM_SAGES_EC_PSD_SEM_DF = ch_CAM_SAGES_EC_PSD_SEM_DF.iloc[cols_freqs[0]]

ch_Ctrl_SAGES_EO_PSD_DF = ch_Ctrl_SAGES_EO_PSD_DF.iloc[cols_freqs[0]]
ch_Ctrl_SAGES_EO_PSD_SEM_DF = ch_Ctrl_SAGES_EO_PSD_SEM_DF.iloc[cols_freqs[0]]

ch_Ctrl_SAGES_EC_PSD_DF = ch_Ctrl_SAGES_EC_PSD_DF.iloc[cols_freqs[0]]
ch_Ctrl_SAGES_EC_PSD_SEM_DF = ch_Ctrl_SAGES_EC_PSD_SEM_DF.iloc[cols_freqs[0]]

freqs_SAGES = np.asarray(freqs_SAGES)[cols_freqs[0]]

# DUKE
freqs_DUKE = list(zip(*(i.split("_") for (i, row) in ch_CAM_DUKE_EO_PSD_DF.items())))[
    -1
]
freqs_flt = [float(f) for f in freqs_DUKE]
freqs_flt_arr = np.asarray(freqs_flt)
cols_freqs = np.where(
    np.logical_and(freqs_flt_arr >= freq_min, freqs_flt_arr <= freq_max)
)

ch_CAM_DUKE_EO_PSD_DF = ch_CAM_DUKE_EO_PSD_DF.iloc[cols_freqs[0]]
ch_CAM_DUKE_EO_PSD_SEM_DF = ch_CAM_DUKE_EO_PSD_SEM_DF.iloc[cols_freqs[0]]

ch_CAM_DUKE_EC_PSD_DF = ch_CAM_DUKE_EC_PSD_DF.iloc[cols_freqs[0]]
ch_CAM_DUKE_EC_PSD_SEM_DF = ch_CAM_DUKE_EC_PSD_SEM_DF.iloc[cols_freqs[0]]

ch_Ctrl_DUKE_EO_PSD_DF = ch_Ctrl_DUKE_EO_PSD_DF.iloc[cols_freqs[0]]
ch_Ctrl_DUKE_EO_PSD_SEM_DF = ch_Ctrl_DUKE_EO_PSD_SEM_DF.iloc[cols_freqs[0]]

ch_Ctrl_DUKE_EC_PSD_DF = ch_Ctrl_DUKE_EC_PSD_DF.iloc[cols_freqs[0]]
ch_Ctrl_DUKE_EC_PSD_SEM_DF = ch_Ctrl_DUKE_EC_PSD_SEM_DF.iloc[cols_freqs[0]]

freqs_DUKE = np.asarray(freqs_DUKE)[cols_freqs[0]]
# freqs_int = [int(float(f)) for f in freqs_DUKE]

fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(24, 8))
fig.suptitle(f"PSDs of {ch_name}", fontsize=20)

# SAGES EO
axs[0].plot(freqs_DUKE, ch_Ctrl_SAGES_EO_PSD_DF, label="Control Mean")
axs[0].plot(freqs_DUKE, ch_CAM_SAGES_EO_PSD_DF, label="Delirium Mean")

axs[0].fill_between(
    freqs_SAGES,
    ch_Ctrl_SAGES_EO_PSD_DF - ch_Ctrl_SAGES_EO_PSD_SEM_DF,
    ch_Ctrl_SAGES_EO_PSD_DF + ch_Ctrl_SAGES_EO_PSD_SEM_DF,
    alpha=0.5,
)
axs[0].fill_between(
    freqs_SAGES,
    ch_CAM_SAGES_EO_PSD_DF - ch_CAM_SAGES_EO_PSD_SEM_DF,
    ch_CAM_SAGES_EO_PSD_DF + ch_CAM_SAGES_EO_PSD_SEM_DF,
    alpha=0.5,
)

# axs[0].legend()
axs[0].set_title(f"SAGES Data Set EO", fontsize=20)
axs[0].set_xlabel("Frequency [Hz]", fontsize=14)
axs[0].set_ylabel("Log Power [$10*log_{10}(\mu V^2/Hz)$]", fontsize=14)
xlocs = axs[0].get_xticks()
xlabels = axs[0].get_xticklabels()
ylocs = axs[0].get_yticks()
ylabels = axs[0].get_yticklabels()
xlabels_int = [int(float(label.get_text())) for label in xlabels]
axs[0].set_xticks(ticks=xlocs[0::6], labels=xlabels_int[0::6], fontsize=14)
axs[0].set_yticks(ticks=ylocs, labels=ylabels, fontsize=12)
# plt.xlim((1,90))
axs[0].grid()

# SAGES EC
axs[1].plot(freqs_SAGES, ch_Ctrl_SAGES_EC_PSD_DF, label="Control Mean")
axs[1].plot(freqs_SAGES, ch_CAM_SAGES_EC_PSD_DF, label="Delirium Mean")

axs[1].fill_between(
    freqs_SAGES,
    ch_Ctrl_SAGES_EC_PSD_DF - ch_Ctrl_SAGES_EC_PSD_SEM_DF,
    ch_Ctrl_SAGES_EC_PSD_DF + ch_Ctrl_SAGES_EC_PSD_SEM_DF,
    alpha=0.5,
)
axs[1].fill_between(
    freqs_SAGES,
    ch_CAM_SAGES_EC_PSD_DF - ch_CAM_SAGES_EC_PSD_SEM_DF,
    ch_CAM_SAGES_EC_PSD_DF + ch_CAM_SAGES_EC_PSD_SEM_DF,
    alpha=0.5,
)

# axs[1].legend()
axs[1].set_title(f"SAGES Data Set EC", fontsize=20)
axs[1].set_xlabel("Frequency [Hz]", fontsize=14)
axs[1].set_ylabel("Log Power [$10*log_{10}(\mu V^2/Hz)$]", fontsize=14)
xlocs = axs[1].get_xticks()
xlabels = axs[1].get_xticklabels()
ylocs = axs[1].get_yticks()
ylabels = axs[1].get_yticklabels()
xlabels_int = [int(float(label.get_text())) for label in xlabels]
axs[1].set_xticks(ticks=xlocs[0::6], labels=xlabels_int[0::6], fontsize=14)
axs[1].set_yticks(ticks=ylocs, labels=ylabels, fontsize=14)
# plt.xlim((1,90))
axs[1].grid()

# DUKE EO
axs[2].plot(freqs_DUKE, ch_Ctrl_DUKE_EO_PSD_DF, label="Control Mean")
axs[2].plot(freqs_DUKE, ch_CAM_DUKE_EO_PSD_DF, label="Delirium Mean")

axs[2].fill_between(
    freqs_DUKE,
    ch_Ctrl_DUKE_EO_PSD_DF - ch_Ctrl_DUKE_EO_PSD_SEM_DF,
    ch_Ctrl_DUKE_EO_PSD_DF + ch_Ctrl_DUKE_EO_PSD_SEM_DF,
    alpha=0.5,
)
axs[2].fill_between(
    freqs_DUKE,
    ch_CAM_DUKE_EO_PSD_DF - ch_CAM_DUKE_EO_PSD_SEM_DF,
    ch_CAM_DUKE_EO_PSD_DF + ch_CAM_DUKE_EO_PSD_SEM_DF,
    alpha=0.5,
)
# axs[2].legend()
axs[2].set_title(f"INTUIT/PRIME Data Set EO", fontsize=20)
axs[2].set_xlabel("Frequency [Hz]", fontsize=14)
axs[2].set_ylabel("Log Power [$10*log_{10}(\mu V^2/Hz)$]", fontsize=14)
xlocs = axs[2].get_xticks()
xlabels = axs[2].get_xticklabels()
ylocs = axs[2].get_yticks()
ylabels = axs[2].get_yticklabels()
xlabels_int = [int(float(label.get_text())) for label in xlabels]
axs[2].set_xticks(ticks=xlocs[0::6], labels=xlabels_int[0::6], fontsize=14)
axs[2].set_yticks(ticks=ylocs, labels=ylabels, fontsize=14)
# plt.xlim((1,90))
axs[2].grid()

# DUKE EC
axs[3].plot(freqs_DUKE, ch_Ctrl_DUKE_EC_PSD_DF, label="Control Mean")
axs[3].plot(freqs_DUKE, ch_CAM_DUKE_EC_PSD_DF, label="Delirium Mean")

axs[3].fill_between(
    freqs_DUKE,
    ch_Ctrl_DUKE_EC_PSD_DF - ch_Ctrl_DUKE_EC_PSD_SEM_DF,
    ch_Ctrl_DUKE_EC_PSD_DF + ch_Ctrl_DUKE_EC_PSD_SEM_DF,
    alpha=0.5,
)
axs[3].fill_between(
    freqs_DUKE,
    ch_CAM_DUKE_EC_PSD_DF - ch_CAM_DUKE_EC_PSD_SEM_DF,
    ch_CAM_DUKE_EC_PSD_DF + ch_CAM_DUKE_EC_PSD_SEM_DF,
    alpha=0.5,
)
axs[3].legend()
axs[3].set_title(f"INTUIT/PRIME Data Set EC", fontsize=20)
axs[3].set_xlabel("Frequency [Hz]", fontsize=14)
axs[3].set_ylabel("Log Power [$10*log_{10}(\mu V^2/Hz)$]", fontsize=14)
xlocs = axs[3].get_xticks()
xlabels = axs[3].get_xticklabels()
ylocs = axs[3].get_yticks()
ylabels = axs[3].get_yticklabels()
xlabels_int = [int(float(label.get_text())) for label in xlabels]
axs[3].set_xticks(ticks=xlocs[0::6], labels=xlabels_int[0::6], fontsize=14)
axs[3].set_yticks(ticks=ylocs, labels=ylabels, fontsize=14)
# plt.xlim((1,90))
axs[3].grid()
plt.tight_layout()

plt.savefig(os.path.join(fig_Dir_SAGES, f"PSD_{ch_name}_Group_Summary.png"))
