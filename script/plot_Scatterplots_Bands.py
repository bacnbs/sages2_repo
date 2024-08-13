"""
Created on Jan 17, 2024

Plot scatterplot of bandpower for each Hz

@author: mning
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from EEGFeaturesExtraction.coreSrc.features import getDeliriumStatusMd


def plot_Scatterplots_Bands(op_SAGES=0, is_EO=0, dist_Medn=2) -> None:
    plt.set_loglevel(level="warning")
    # for debugging purpose
    # c4_str = 'C4'
    # very slow!!!
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

    if op_SAGES:
        figDir = r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data\Baseline_rsEEG_preprocessing\DATA_visualization\Group\PSDs"
        featFN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_BP_Bands_{eo_str}_{relative_str}_car_mV.csv"
        set_name = "SAGES"
        title_name = "SAGES"
    else:
        figDir = r"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA\DATA_visualization\Group\PSDs"
        featFN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj_BP_Bands_{eo_str}_{relative_str}_car_mV.csv"
        set_name = "INTUIT_PRIME"
        title_name = "INTUIT/PRIME"

    feats_DF = pd.read_csv(
        featFN,
        dtype={
            "# SbjID": pd.StringDtype(),
            "TrialNum": np.int32,
            "SetFile": pd.StringDtype(),
        },
    )
    # feats_bipolar_DF = pd.read_csv(featFN_bipolar, dtype={'# SbjID': pd.StringDtype(), 'TrialNum': np.int32, 'SetFile': pd.StringDtype()})

    # Compute SPR first before normalizing
    # psd_df = feats_DF.filter(regex=f"(# SbjID)|(TrialNum)|(^{ch_name}_)", axis="columns")
    new_psd_df = feats_DF.drop("SetFile", axis="columns", inplace=False)

    del feats_DF

    new_psd_df.set_index(["# SbjID", "TrialNum"], inplace=True)

    if dist_Medn == 1:
        # distance to the median
        # spr_transform_df = spr_df.apply(lambda x: np.log10(np.where(np.abs(x-x.median())!=0,np.abs(x-x.median()),np.NaN)))
        # abr_transform_df = abr_df.apply(lambda x: np.log10(np.where(np.abs(x-x.median())!=0,np.abs(x-x.median()),np.NaN)))
        # spr_max_transform_df = spr_max_df.apply(lambda x: np.log10(np.where(np.abs(x-x.median())!=0,np.abs(x-x.median()),np.NaN)))
        # spr_transform_df = spr_df.apply(lambda x: np.log10(np.abs(x-x.median()+0.1)))
        # abr_transform_df = abr_df.apply(lambda x: np.log10(np.abs(x-x.median()+0.1)))
        # spr_max_transform_df = spr_max_df.apply(lambda x: np.log10(np.abs(x-x.median()+0.1)))
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
        new_psd_transform_df.replace(np.NaN, 0, inplace=True)
        # spr_transform_df = spr_df.apply(lambda x: np.log10)
        # spr_transform_df = spr_df.apply(lambda x: np.abs((x-x.median())))
    elif dist_Medn == 3:
        whole_medn = np.log10(np.median(new_psd_df))
        new_psd_transform_df = new_psd_df.apply(
            lambda x: np.where(
                np.abs(np.log10(x) - whole_medn) != 0,
                np.abs(np.log10(x) - whole_medn),
                np.NaN,
            )
        )
        new_psd_transform_df.replace(np.NaN, 0, inplace=True)
    elif dist_Medn == 0:
        new_psd_transform_df = new_psd_df.apply(scipy.stats.zscore)
    elif dist_Medn == 6:
        new_psd_transform_df = new_psd_df.apply(lambda x: 10 * np.log10(x))

    CAM = getDeliriumStatusMd.getDeliriumStatus(op_SAGES)

    feats_DF_CAM_Sbj = new_psd_transform_df[
        new_psd_transform_df.index.isin(CAM, level=0)
    ]
    feats_DF_Ctrl_Sbj = new_psd_transform_df[
        ~new_psd_transform_df.index.isin(CAM, level=0)
    ]

    feats_DF_CAM_Grp = feats_DF_CAM_Sbj.mean(axis="rows").to_frame().T
    feats_DF_Ctrl_Grp = feats_DF_Ctrl_Sbj.mean(axis="rows").to_frame().T

    if op_SAGES:
        sel_ch_names_Delta = [
            "O1_delta",
            "O2_delta",
            "POz_delta",
            "Oz_delta",
            "PO3_delta",
            "PO4_delta",
        ]

        sel_ch_names_Theta = [
            "O1_theta",
            "O2_theta",
            "POz_theta",
            "Oz_theta",
            "PO3_theta",
            "PO4_theta",
        ]

        sel_ch_names_Alpha = [
            "O1_alpha",
            "O2_alpha",
            "POz_alpha",
            "Oz_alpha",
            "PO3_alpha",
            "PO4_alpha",
        ]

        sel_ch_names_Beta = [
            "O1_beta",
            "O2_beta",
            "POz_beta",
            "Oz_beta",
            "PO3_beta",
            "PO4_beta",
        ]
    else:
        sel_ch_names_Delta = [
            "O1_delta",
            "O2_delta",
            "POz_delta",
        ]

        sel_ch_names_Theta = [
            "O1_theta",
            "O2_theta",
            "POz_theta",
        ]

        sel_ch_names_Alpha = [
            "O1_alpha",
            "O2_alpha",
            "POz_alpha",
        ]

        sel_ch_names_Beta = [
            "O1_beta",
            "O2_beta",
            "POz_beta",
        ]

    id_O1_alpha = "O1_alpha"
    id_O2_alpha = "O2_alpha"
    id_Oz_alpha = "Oz_alpha"
    id_POz_alpha = "POz_alpha"
    id_PO3_alpha = "PO3_alpha"
    id_PO4_alpha = "PO4_alpha"

    id_O1_beta = "O1_beta"
    id_O2_beta = "O2_beta"
    id_Oz_beta = "Oz_beta"
    id_POz_beta = "POz_beta"
    id_PO3_beta = "PO3_beta"
    id_PO4_beta = "PO4_beta"

    id_O1_delta = "O1_delta"
    id_O2_delta = "O2_delta"
    id_Oz_delta = "Oz_delta"
    id_POz_delta = "POz_delta"
    id_PO3_delta = "PO3_delta"
    id_PO4_delta = "PO4_delta"

    id_O1_theta = "O1_theta"
    id_O2_theta = "O2_theta"
    id_Oz_theta = "Oz_theta"
    id_POz_theta = "POz_theta"
    id_PO3_theta = "PO3_theta"
    id_PO4_theta = "PO4_theta"

    if op_SAGES:
        x_tick_idcs = [0, 1, 2, 3, 4, 5]
    else:
        x_tick_idcs = [0, 1, 2]

    if op_SAGES:
        ttst_O1 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_theta],
            equal_var=False,
        )
        ttst_O2 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_theta],
            equal_var=False,
        )
        ttst_POz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_theta],
            equal_var=False,
        )
        ttst_Oz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_Oz_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_theta],
            equal_var=False,
        )
        ttst_PO3 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO3_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_theta],
            equal_var=False,
        )
        ttst_PO4 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO4_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_theta],
            equal_var=False,
        )

        pval_list = [
            ttst_O1.pvalue,
            ttst_O2.pvalue,
            ttst_POz.pvalue,
            ttst_Oz.pvalue,
            ttst_PO3.pvalue,
            ttst_PO4.pvalue,
        ]

    else:
        ttst_O1 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_theta],
            equal_var=False,
        )
        ttst_O2 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_theta],
            equal_var=False,
        )
        ttst_POz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_theta],
            equal_var=False,
        )

        pval_list = [
            ttst_O1.pvalue,
            ttst_O2.pvalue,
            ttst_POz.pvalue,
        ]

    x_ticks_lbls = [
        chn_name + " (" + f"{pval:.2f})"
        for (chn_name, pval) in zip(sel_ch_names_Theta, pval_list)
    ]

    fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
    fig.suptitle(rf"Theta PSD {title_name} {eo_str} {scale_str}")

    plt.plot(
        -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_O1_theta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_O2_theta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_POz_theta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        -0.2,
        feats_DF_Ctrl_Grp.loc[0, id_O1_theta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        0.8,
        feats_DF_Ctrl_Grp.loc[0, id_O2_theta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        1.8,
        feats_DF_Ctrl_Grp.loc[0, id_POz_theta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )

    plt.plot(
        0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_O1_theta],
        marker="o",
        color="m",
        linestyle="None",
    )
    plt.plot(
        1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_O2_theta],
        marker="o",
        color="m",
        linestyle="None",
    )
    plt.plot(
        2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_POz_theta],
        marker="o",
        color="m",
        linestyle="None",
    )

    plt.plot(
        0.2,
        feats_DF_CAM_Grp.loc[0, id_O1_theta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        1.2,
        feats_DF_CAM_Grp.loc[0, id_O2_theta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        2.2,
        feats_DF_CAM_Grp.loc[0, id_POz_theta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )

    if op_SAGES:
        plt.plot(
            2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_theta],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_Oz_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_theta],
            equal_var=False,
        )

        plt.plot(
            3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_theta],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO3_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_theta],
            equal_var=False,
        )

        plt.plot(
            4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_theta],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO4_theta],
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_theta],
            equal_var=False,
        )

        plt.plot(
            2.8,
            feats_DF_Ctrl_Grp.loc[0, id_Oz_theta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            3.8,
            feats_DF_Ctrl_Grp.loc[0, id_PO3_theta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            4.8,
            feats_DF_Ctrl_Grp.loc[0, id_PO4_theta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_Oz_theta],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_PO3_theta],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_PO4_theta],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            3.2,
            feats_DF_CAM_Grp.loc[0, id_Oz_theta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            4.2,
            feats_DF_CAM_Grp.loc[0, id_PO3_theta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            5.2,
            feats_DF_CAM_Grp.loc[0, id_PO4_theta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

    plt.xticks(x_tick_idcs, x_ticks_lbls, rotation=45)
    # plt.ylim(0,2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figDir, f"ScatterPlot_Theta_BP_{set_name}_{eo_str}_{scale_str}.png"
        )
    )

    if op_SAGES:
        ttst_O1 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_alpha],
            equal_var=False,
        )
        ttst_O2 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_alpha],
            equal_var=False,
        )
        ttst_POz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_alpha],
            equal_var=False,
        )
        ttst_Oz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_Oz_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_alpha],
            equal_var=False,
        )
        ttst_PO3 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO3_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_alpha],
            equal_var=False,
        )
        ttst_PO4 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO4_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_alpha],
            equal_var=False,
        )

        pval_list = [
            ttst_O1.pvalue,
            ttst_O2.pvalue,
            ttst_POz.pvalue,
            ttst_Oz.pvalue,
            ttst_PO3.pvalue,
            ttst_PO4.pvalue,
        ]

    else:
        ttst_O1 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_alpha],
            equal_var=False,
        )
        ttst_O2 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_alpha],
            equal_var=False,
        )
        ttst_POz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_alpha],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_alpha],
            equal_var=False,
        )

        pval_list = [
            ttst_O1.pvalue,
            ttst_O2.pvalue,
            ttst_POz.pvalue,
        ]

    x_ticks_lbls = [
        chn_name + " (" + f"{pval:.2f})"
        for (chn_name, pval) in zip(sel_ch_names_Alpha, pval_list)
    ]

    fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
    fig.suptitle(rf"Alpha {title_name} {eo_str} {scale_str}")

    plt.plot(
        -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_O1_alpha],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_O2_alpha],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_POz_alpha],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        -0.2,
        feats_DF_Ctrl_Grp.loc[0, id_O1_alpha],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        0.8,
        feats_DF_Ctrl_Grp.loc[0, id_O2_alpha],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        1.8,
        feats_DF_Ctrl_Grp.loc[0, id_POz_alpha],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )

    plt.plot(
        0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_O1_alpha],
        marker="o",
        color="m",
        linestyle="None",
    )
    plt.plot(
        1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_O2_alpha],
        marker="o",
        color="m",
        linestyle="None",
    )
    plt.plot(
        2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_POz_alpha],
        marker="o",
        color="m",
        linestyle="None",
    )

    plt.plot(
        0.2,
        feats_DF_CAM_Grp.loc[0, id_O1_alpha],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        1.2,
        feats_DF_CAM_Grp.loc[0, id_O2_alpha],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        2.2,
        feats_DF_CAM_Grp.loc[0, id_POz_alpha],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )

    if op_SAGES:
        plt.plot(
            2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_alpha],
            marker="o",
            color="b",
            linestyle="None",
        )

        plt.plot(
            3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_alpha],
            marker="o",
            color="b",
            linestyle="None",
        )

        plt.plot(
            4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_alpha],
            marker="o",
            color="b",
            linestyle="None",
        )

        plt.plot(
            2.8,
            feats_DF_Ctrl_Grp.loc[0, id_Oz_alpha],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            3.8,
            feats_DF_Ctrl_Grp.loc[0, id_PO3_alpha],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            4.8,
            feats_DF_Ctrl_Grp.loc[0, id_PO4_alpha],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_Oz_alpha],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_PO3_alpha],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_PO4_alpha],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            3.2,
            feats_DF_CAM_Grp.loc[0, id_Oz_alpha],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            4.2,
            feats_DF_CAM_Grp.loc[0, id_PO3_alpha],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            5.2,
            feats_DF_CAM_Grp.loc[0, id_PO4_alpha],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

    plt.xticks(x_tick_idcs, x_ticks_lbls, rotation=45)
    # plt.ylim(0,2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figDir, f"ScatterPlot_Alpha_BP_{set_name}_{eo_str}_{scale_str}.png"
        )
    )

    if op_SAGES:
        ttst_O1 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_beta],
            equal_var=False,
        )
        ttst_O2 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_beta],
            equal_var=False,
        )
        ttst_POz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_beta],
            equal_var=False,
        )
        ttst_Oz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_Oz_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_beta],
            equal_var=False,
        )
        ttst_PO3 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO3_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_beta],
            equal_var=False,
        )
        ttst_PO4 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO4_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_beta],
            equal_var=False,
        )

        pval_list = [
            ttst_O1.pvalue,
            ttst_O2.pvalue,
            ttst_POz.pvalue,
            ttst_Oz.pvalue,
            ttst_PO3.pvalue,
            ttst_PO4.pvalue,
        ]

    else:
        ttst_O1 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_beta],
            equal_var=False,
        )
        ttst_O2 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_beta],
            equal_var=False,
        )
        ttst_POz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_beta],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_beta],
            equal_var=False,
        )

        pval_list = [
            ttst_O1.pvalue,
            ttst_O2.pvalue,
            ttst_POz.pvalue,
        ]

    x_ticks_lbls = [
        chn_name + " (" + f"{pval:.2f})"
        for (chn_name, pval) in zip(sel_ch_names_Beta, pval_list)
    ]

    fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
    fig.suptitle(rf"Beta PSD {title_name} {eo_str} {scale_str}")

    plt.plot(
        -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_O1_beta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_O2_beta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_POz_beta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        -0.2,
        feats_DF_Ctrl_Grp.loc[0, id_O1_beta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        0.8,
        feats_DF_Ctrl_Grp.loc[0, id_O2_beta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        1.8,
        feats_DF_Ctrl_Grp.loc[0, id_POz_beta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )

    plt.plot(
        0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_O1_beta],
        marker="o",
        color="m",
        linestyle="None",
    )
    plt.plot(
        1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_O2_beta],
        marker="o",
        color="m",
        linestyle="None",
    )
    plt.plot(
        2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_POz_beta],
        marker="o",
        color="m",
        linestyle="None",
    )

    plt.plot(
        0.2,
        feats_DF_CAM_Grp.loc[0, id_O1_beta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        1.2,
        feats_DF_CAM_Grp.loc[0, id_O2_beta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        2.2,
        feats_DF_CAM_Grp.loc[0, id_POz_beta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )

    if op_SAGES:
        plt.plot(
            2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_beta],
            marker="o",
            color="b",
            linestyle="None",
        )

        plt.plot(
            3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_beta],
            marker="o",
            color="b",
            linestyle="None",
        )

        plt.plot(
            4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_beta],
            marker="o",
            color="b",
            linestyle="None",
        )

        plt.plot(
            2.8,
            feats_DF_Ctrl_Grp.loc[0, id_Oz_beta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            3.8,
            feats_DF_Ctrl_Grp.loc[0, id_PO3_beta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            4.8,
            feats_DF_Ctrl_Grp.loc[0, id_PO4_beta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_Oz_beta],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_PO3_beta],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_PO4_beta],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            3.2,
            feats_DF_CAM_Grp.loc[0, id_Oz_beta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            4.2,
            feats_DF_CAM_Grp.loc[0, id_PO3_beta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            5.2,
            feats_DF_CAM_Grp.loc[0, id_PO4_beta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

    plt.xticks(x_tick_idcs, x_ticks_lbls, rotation=45)
    # plt.ylim(0,2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(figDir, f"ScatterPlot_Beta_BP_{set_name}_{eo_str}_{scale_str}.png")
    )

    if op_SAGES:
        ttst_O1 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_delta],
            equal_var=False,
        )
        ttst_O2 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_delta],
            equal_var=False,
        )
        ttst_POz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_delta],
            equal_var=False,
        )
        ttst_Oz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_Oz_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_delta],
            equal_var=False,
        )
        ttst_PO3 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO3_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_delta],
            equal_var=False,
        )
        ttst_PO4 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_PO4_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_delta],
            equal_var=False,
        )

        pval_list = [
            ttst_O1.pvalue,
            ttst_O2.pvalue,
            ttst_POz.pvalue,
            ttst_Oz.pvalue,
            ttst_PO3.pvalue,
            ttst_PO4.pvalue,
        ]

    else:
        ttst_O1 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_delta],
            equal_var=False,
        )
        ttst_O2 = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_delta],
            equal_var=False,
        )
        ttst_POz = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_delta],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_delta],
            equal_var=False,
        )

        pval_list = [
            ttst_O1.pvalue,
            ttst_O2.pvalue,
            ttst_POz.pvalue,
        ]

    x_ticks_lbls = [
        chn_name + " (" + f"{pval:.2f})"
        for (chn_name, pval) in zip(sel_ch_names_Delta, pval_list)
    ]

    fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
    fig.suptitle(rf"Delta PSD {title_name} {eo_str} {scale_str}")

    plt.plot(
        -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_O1_delta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_O2_delta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
        feats_DF_Ctrl_Sbj.loc[:, id_POz_delta],
        marker="o",
        color="b",
        linestyle="None",
    )

    plt.plot(
        -0.2,
        feats_DF_Ctrl_Grp.loc[0, id_O1_delta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        0.8,
        feats_DF_Ctrl_Grp.loc[0, id_O2_delta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        1.8,
        feats_DF_Ctrl_Grp.loc[0, id_POz_delta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )

    plt.plot(
        0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_O1_delta],
        marker="o",
        color="m",
        linestyle="None",
    )
    plt.plot(
        1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_O2_delta],
        marker="o",
        color="m",
        linestyle="None",
    )
    plt.plot(
        2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
        feats_DF_CAM_Sbj.loc[:, id_POz_delta],
        marker="o",
        color="m",
        linestyle="None",
    )

    plt.plot(
        0.2,
        feats_DF_CAM_Grp.loc[0, id_O1_delta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        1.2,
        feats_DF_CAM_Grp.loc[0, id_O2_delta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )
    plt.plot(
        2.2,
        feats_DF_CAM_Grp.loc[0, id_POz_delta],
        marker="s",
        markersize=10,
        color="k",
        linestyle="None",
    )

    if op_SAGES:
        plt.plot(
            2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_Oz_delta],
            marker="o",
            color="b",
            linestyle="None",
        )

        plt.plot(
            3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_PO3_delta],
            marker="o",
            color="b",
            linestyle="None",
        )

        plt.plot(
            4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_PO4_delta],
            marker="o",
            color="b",
            linestyle="None",
        )
        plt.plot(
            2.8,
            feats_DF_Ctrl_Grp.loc[0, id_Oz_delta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            3.8,
            feats_DF_Ctrl_Grp.loc[0, id_PO3_delta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            4.8,
            feats_DF_Ctrl_Grp.loc[0, id_PO4_delta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_Oz_delta],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_PO3_delta],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_PO4_delta],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            3.2,
            feats_DF_CAM_Grp.loc[0, id_Oz_delta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            4.2,
            feats_DF_CAM_Grp.loc[0, id_PO3_delta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            5.2,
            feats_DF_CAM_Grp.loc[0, id_PO4_delta],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

    plt.xticks(x_tick_idcs, x_ticks_lbls, rotation=45)
    # plt.ylim(0,2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            figDir, f"ScatterPlot_Delta_BP_{set_name}_{eo_str}_{scale_str}.png"
        )
    )

    plt.close("all")
