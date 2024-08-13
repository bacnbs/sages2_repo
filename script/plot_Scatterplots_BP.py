"""
Created on Sep 25, 2023

Plot scatterplot of bandpower for each Hz

@author: mning

12/13/2023
Add support for Duke dataset
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from EEGFeaturesExtraction.coreSrc.features import getDeliriumStatusMd


def plot_Scatterplots_BP(op_SAGES=0, is_EO=0, dist_Medn=2) -> None:
    plt.set_loglevel(level="warning")
    # dist_Medn only applies to SPR and not cognitive assessment scores
    binWidth = 2
    bin2_startF = 0
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
    if binWidth == 2 and not bin2_startF:
        binStr = "2EvenHz"
    elif binWidth == 2 and bin2_startF:
        binStr = "2Hz"
    else:
        binStr = "1Hz"

    if op_SAGES and binWidth == 2 and not bin2_startF:
        figDir = r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data\Baseline_rsEEG_preprocessing\DATA_visualization\Group\PSDs"
        featFN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_BP_{binWidth}EvenHz_{eo_str}_{relative_str}_car_mV.csv"
        set_name = "SAGES"
        title_name = "SAGES"
    elif op_SAGES:
        figDir = r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data\Baseline_rsEEG_preprocessing\DATA_visualization\Group\PSDs"
        featFN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Data\Baseline_rsEEG_preprocessing\FeatureStores\featureStore_Sbj_BP_{binWidth}Hz_{eo_str}_{relative_str}_car_mV.csv"
        set_name = "SAGES"
        title_name = "SAGES"
    elif not op_SAGES and binWidth == 2 and not bin2_startF:
        figDir = r"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA\DATA_visualization\Group\PSDs"
        featFN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj_BP_{binWidth}EvenHz_{eo_str}_{relative_str}_car_mV.csv"
        set_name = "INTUIT_PRIME"
        title_name = "INTUIT/PRIME"
    else:
        figDir = r"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA\DATA_visualization\Group\PSDs"
        featFN = rf"R:\Studies_Investigator Folders\Shafi\2018P-000227_SAGES I and II\Duke Preop EEG Data\DATA\FeatureStores\featureStore_Sbj_BP_{binWidth}Hz_{eo_str}_{relative_str}_car_mV.csv"
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
        sel_ch_names_4Hz = [
            "O1_4",
            "O2_4",
            "POz_4",
            "Oz_4",
            "PO3_4",
            "PO4_4",
        ]

        sel_ch_names_5Hz = [
            "O1_5",
            "O2_5",
            "POz_5",
            "Oz_5",
            "PO3_5",
            "PO4_5",
        ]

        sel_ch_names_6Hz = [
            "O1_6",
            "O2_6",
            "POz_6",
            "Oz_6",
            "PO3_6",
            "PO4_6",
        ]

        sel_ch_names_8Hz = [
            "O1_8",
            "O2_8",
            "POz_8",
            "Oz_8",
            "PO3_8",
            "PO4_8",
        ]

        sel_ch_names_9Hz = [
            "O1_9",
            "O2_9",
            "POz_9",
            "Oz_9",
            "PO3_9",
            "PO4_9",
        ]

        sel_ch_names_10Hz = [
            "O1_10",
            "O2_10",
            "POz_10",
            "Oz_10",
            "PO3_10",
            "PO4_10",
        ]

        sel_ch_names_11Hz = [
            "O1_11",
            "O2_11",
            "POz_11",
            "Oz_11",
            "PO3_11",
            "PO4_11",
        ]

        sel_ch_names_12Hz = [
            "O1_12",
            "O2_12",
            "POz_12",
            "Oz_12",
            "PO3_12",
            "PO4_12",
        ]
    else:
        sel_ch_names_4Hz = [
            "O1_4",
            "O2_4",
            "POz_4",
        ]

        sel_ch_names_5Hz = [
            "O1_5",
            "O2_5",
            "POz_5",
        ]

        sel_ch_names_6Hz = [
            "O1_6",
            "O2_6",
            "POz_6",
        ]

        sel_ch_names_8Hz = [
            "O1_8",
            "O2_8",
            "POz_8",
        ]

        sel_ch_names_9Hz = [
            "O1_9",
            "O2_9",
            "POz_9",
        ]

        sel_ch_names_10Hz = [
            "O1_10",
            "O2_10",
            "POz_10",
        ]

        sel_ch_names_11Hz = [
            "O1_11",
            "O2_11",
            "POz_11",
        ]

        sel_ch_names_12Hz = [
            "O1_12",
            "O2_12",
            "POz_12",
        ]

    # sel_ch_names = ['O1_11',
    #                 'O2_11',
    #                 'Oz_11',
    #                 'POz_11',
    #                 'PO3_11',
    #                 'PO4_11',
    #                 'O1_9',
    #                 'O2_9',
    #                 'Oz_9',
    #                 'POz_9',
    #                 'PO3_9',
    #                 'PO4_9',]

    id_O1_4 = "O1_4"
    id_O2_4 = "O2_4"
    id_Oz_4 = "Oz_4"
    id_POz_4 = "POz_4"
    id_PO3_4 = "PO3_4"
    id_PO4_4 = "PO4_4"

    id_O1_5 = "O1_5"
    id_O2_5 = "O2_5"
    id_Oz_5 = "Oz_5"
    id_POz_5 = "POz_5"
    id_PO3_5 = "PO3_5"
    id_PO4_5 = "PO4_5"

    id_O1_6 = "O1_6"
    id_O2_6 = "O2_6"
    id_Oz_6 = "Oz_6"
    id_POz_6 = "POz_6"
    id_PO3_6 = "PO3_6"
    id_PO4_6 = "PO4_6"

    id_O1_12 = "O1_12"
    id_O2_12 = "O2_12"
    id_Oz_12 = "Oz_12"
    id_POz_12 = "POz_12"
    id_PO3_12 = "PO3_12"
    id_PO4_12 = "PO4_12"

    id_O1_11 = "O1_11"
    id_O2_11 = "O2_11"
    id_Oz_11 = "Oz_11"
    id_POz_11 = "POz_11"
    id_PO3_11 = "PO3_11"
    id_PO4_11 = "PO4_11"

    id_O1_10 = "O1_10"
    id_O2_10 = "O2_10"
    id_Oz_10 = "Oz_10"
    id_POz_10 = "POz_10"
    id_PO3_10 = "PO3_10"
    id_PO4_10 = "PO4_10"

    id_O1_9 = "O1_9"
    id_O2_9 = "O2_9"
    id_Oz_9 = "Oz_9"
    id_POz_9 = "POz_9"
    id_PO3_9 = "PO3_9"
    id_PO4_9 = "PO4_9"

    id_O1_8 = "O1_8"
    id_O2_8 = "O2_8"
    id_Oz_8 = "Oz_8"
    id_POz_8 = "POz_8"
    id_PO3_8 = "PO3_8"
    id_PO4_8 = "PO4_8"

    if op_SAGES:
        x_tick_idcs = [0, 1, 2, 3, 4, 5]
    else:
        x_tick_idcs = [0, 1, 2]

    if (binWidth == 2 and not bin2_startF) or binWidth == 1:
        if op_SAGES:
            ttst_O1 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O1_8],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_8],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_8],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_8],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_8],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_8],
                equal_var=False,
            )
            ttst_Oz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_8],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_8],
                equal_var=False,
            )
            ttst_PO3 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_8],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_8],
                equal_var=False,
            )
            ttst_PO4 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_8],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_8],
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
                feats_DF_CAM_Sbj.loc[:, id_O1_8],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_8],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_8],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_8],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_8],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_8],
                equal_var=False,
            )

            pval_list = [
                ttst_O1.pvalue,
                ttst_O2.pvalue,
                ttst_POz.pvalue,
            ]

        x_ticks_lbls = [
            chn_name + " (" + f"{pval:.2f}" + ")"
            for (chn_name, pval) in zip(sel_ch_names_8Hz, pval_list)
        ]

        fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        fig.suptitle(rf"8 Hz ({binStr} Bin-Avg.) PSD {title_name} {eo_str} {scale_str}")
        # axs[0].tick_params(axis="x", bottom=True, top=True, labelbottom=True, labeltop=True)

        plt.plot(
            -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O1_8],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_8],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_8],
            equal_var=False,
        )
        # plt.text(0,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O2_8],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_8],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_8],
            equal_var=False,
        )
        # plt.text(1,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_POz_8],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_8],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_8],
            equal_var=False,
        )
        # plt.text(2,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            -0.2,
            feats_DF_Ctrl_Grp.loc[0, id_O1_8],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            0.8,
            feats_DF_Ctrl_Grp.loc[0, id_O2_8],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.8,
            feats_DF_Ctrl_Grp.loc[0, id_POz_8],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O1_8],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O2_8],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_POz_8],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            0.2,
            feats_DF_CAM_Grp.loc[0, id_O1_8],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.2,
            feats_DF_CAM_Grp.loc[0, id_O2_8],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            2.2,
            feats_DF_CAM_Grp.loc[0, id_POz_8],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        if op_SAGES:
            plt.plot(
                2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_8],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_8],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_8],
                equal_var=False,
            )
            # plt.text(3,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_8],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_8],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_8],
                equal_var=False,
            )
            # plt.text(4,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_8],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_8],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_8],
                equal_var=False,
            )
            # plt.text(5,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                2.8,
                feats_DF_Ctrl_Grp.loc[0, id_Oz_8],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                3.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO3_8],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO4_8],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_Oz_8],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO3_8],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO4_8],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                3.2,
                feats_DF_CAM_Grp.loc[0, id_Oz_8],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                4.2,
                feats_DF_CAM_Grp.loc[0, id_PO3_8],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                5.2,
                feats_DF_CAM_Grp.loc[0, id_PO4_8],
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
                figDir,
                f"ScatterPlot_8Hz_BP_{binStr}_{set_name}_{eo_str}_{scale_str}.png",
            )
        )

    if (binWidth == 2 and bin2_startF) or binWidth == 1:
        if op_SAGES:
            ttst_O1 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O1_9],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_9],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_9],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_9],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_9],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_9],
                equal_var=False,
            )
            ttst_Oz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_9],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_9],
                equal_var=False,
            )
            ttst_PO3 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_9],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_9],
                equal_var=False,
            )
            ttst_PO4 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_9],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_9],
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
                feats_DF_CAM_Sbj.loc[:, id_O1_9],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_9],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_9],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_9],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_9],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_9],
                equal_var=False,
            )

            pval_list = [
                ttst_O1.pvalue,
                ttst_O2.pvalue,
                ttst_POz.pvalue,
            ]

        x_ticks_lbls = [
            chn_name + " (" + f"{pval:.2f}" + ")"
            for (chn_name, pval) in zip(sel_ch_names_9Hz, pval_list)
        ]

        fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        fig.suptitle(rf"9 Hz ({binStr} Bin-Avg.) PSD {title_name} {eo_str} {scale_str}")

        plt.plot(
            -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O1_9],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_9],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_9],
            equal_var=False,
        )
        # plt.text(0,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O2_9],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_9],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_9],
            equal_var=False,
        )
        # plt.text(1,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_POz_9],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_9],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_9],
            equal_var=False,
        )
        # plt.text(2,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            -0.2,
            feats_DF_Ctrl_Grp.loc[0, id_O1_9],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            0.8,
            feats_DF_Ctrl_Grp.loc[0, id_O2_9],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.8,
            feats_DF_Ctrl_Grp.loc[0, id_POz_9],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O1_9],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O2_9],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_POz_9],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            0.2,
            feats_DF_CAM_Grp.loc[0, id_O1_9],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.2,
            feats_DF_CAM_Grp.loc[0, id_O2_9],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            2.2,
            feats_DF_CAM_Grp.loc[0, id_POz_9],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        if op_SAGES:
            plt.plot(
                2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_9],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_9],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_9],
                equal_var=False,
            )
            # plt.text(3,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_9],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_9],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_9],
                equal_var=False,
            )
            # plt.text(4,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_9],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_9],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_9],
                equal_var=False,
            )
            # plt.text(5,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                2.8,
                feats_DF_Ctrl_Grp.loc[0, id_Oz_9],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO3_9],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO4_9],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_Oz_9],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO3_9],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO4_9],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                3.2,
                feats_DF_CAM_Grp.loc[0, id_Oz_9],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                4.2,
                feats_DF_CAM_Grp.loc[0, id_PO3_9],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                5.2,
                feats_DF_CAM_Grp.loc[0, id_PO4_9],
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
                figDir,
                f"ScatterPlot_9Hz_BP_{binStr}_{set_name}_{eo_str}_{scale_str}.png",
            )
        )

    if (binWidth == 2 and not bin2_startF) or binWidth == 1:
        if op_SAGES:
            ttst_O1 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O1_10],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_10],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_10],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_10],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_10],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_10],
                equal_var=False,
            )
            ttst_Oz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_10],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_10],
                equal_var=False,
            )
            ttst_PO3 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_10],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_10],
                equal_var=False,
            )
            ttst_PO4 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_10],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_10],
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
                feats_DF_CAM_Sbj.loc[:, id_O1_10],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_10],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_10],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_10],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_10],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_10],
                equal_var=False,
            )

            pval_list = [
                ttst_O1.pvalue,
                ttst_O2.pvalue,
                ttst_POz.pvalue,
            ]

        x_ticks_lbls = [
            chn_name + " (" + f"{pval:.2f})"
            for (chn_name, pval) in zip(sel_ch_names_10Hz, pval_list)
        ]

        fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        fig.suptitle(
            rf"10 Hz ({binStr} Bin-Avg.) PSD {title_name} {eo_str} {scale_str}"
        )

        plt.plot(
            -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O1_10],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_10],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_10],
            equal_var=False,
        )
        # plt.text(0,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O2_10],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_10],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_10],
            equal_var=False,
        )
        # plt.text(1,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_POz_10],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_10],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_10],
            equal_var=False,
        )
        # plt.text(2,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            -0.2,
            feats_DF_Ctrl_Grp.loc[0, id_O1_10],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            0.8,
            feats_DF_Ctrl_Grp.loc[0, id_O2_10],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.8,
            feats_DF_Ctrl_Grp.loc[0, id_POz_10],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O1_10],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O2_10],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_POz_10],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            0.2,
            feats_DF_CAM_Grp.loc[0, id_O1_10],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.2,
            feats_DF_CAM_Grp.loc[0, id_O2_10],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            2.2,
            feats_DF_CAM_Grp.loc[0, id_POz_10],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        if op_SAGES:
            plt.plot(
                2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_10],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_10],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_10],
                equal_var=False,
            )
            # plt.text(3,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_10],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_10],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_10],
                equal_var=False,
            )
            # plt.text(4,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_10],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_10],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_10],
                equal_var=False,
            )
            # plt.text(5,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                2.8,
                feats_DF_Ctrl_Grp.loc[0, id_Oz_10],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO3_10],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO4_10],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_Oz_10],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO3_10],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO4_10],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                3.2,
                feats_DF_CAM_Grp.loc[0, id_Oz_10],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.2,
                feats_DF_CAM_Grp.loc[0, id_PO3_10],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                5.2,
                feats_DF_CAM_Grp.loc[0, id_PO4_10],
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
                figDir,
                f"ScatterPlot_10Hz_BP_{binStr}_{set_name}_{eo_str}_{scale_str}.png",
            )
        )

    if (binWidth == 2 and bin2_startF) or binWidth == 1:
        if op_SAGES:
            ttst_O1 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O1_11],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_11],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_11],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_11],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_11],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_11],
                equal_var=False,
            )
            ttst_Oz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_11],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_11],
                equal_var=False,
            )
            ttst_PO3 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_11],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_11],
                equal_var=False,
            )
            ttst_PO4 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_11],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_11],
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
                feats_DF_CAM_Sbj.loc[:, id_O1_11],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_11],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_11],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_11],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_11],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_11],
                equal_var=False,
            )

            pval_list = [
                ttst_O1.pvalue,
                ttst_O2.pvalue,
                ttst_POz.pvalue,
            ]

        x_ticks_lbls = [
            chn_name + " (" + f"{pval:.2f})"
            for (chn_name, pval) in zip(sel_ch_names_11Hz, pval_list)
        ]

        fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        fig.suptitle(
            rf"11 Hz ({binStr} Bin-Avg.) PSD {title_name} {eo_str} {scale_str}"
        )

        plt.plot(
            -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O1_11],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_11],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_11],
            equal_var=False,
        )
        # plt.text(0,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O2_11],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_11],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_11],
            equal_var=False,
        )
        # plt.text(1,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_POz_11],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_11],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_11],
            equal_var=False,
        )
        # plt.text(2,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            -0.2,
            feats_DF_Ctrl_Grp.loc[0, id_O1_11],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            0.8,
            feats_DF_Ctrl_Grp.loc[0, id_O2_11],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.8,
            feats_DF_Ctrl_Grp.loc[0, id_POz_11],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O1_11],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O2_11],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_POz_11],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            0.2,
            feats_DF_CAM_Grp.loc[0, id_O1_11],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.2,
            feats_DF_CAM_Grp.loc[0, id_O2_11],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            2.2,
            feats_DF_CAM_Grp.loc[0, id_POz_11],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        if op_SAGES:
            plt.plot(
                2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_11],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_11],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_11],
                equal_var=False,
            )
            # plt.text(3,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_11],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_11],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_11],
                equal_var=False,
            )
            # plt.text(4,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_11],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_11],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_11],
                equal_var=False,
            )
            # plt.text(5,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                2.8,
                feats_DF_Ctrl_Grp.loc[0, id_Oz_11],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO3_11],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO4_11],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_Oz_11],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO3_11],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO4_11],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                3.2,
                feats_DF_CAM_Grp.loc[0, id_Oz_11],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                4.2,
                feats_DF_CAM_Grp.loc[0, id_PO3_11],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                5.2,
                feats_DF_CAM_Grp.loc[0, id_PO4_11],
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
                figDir,
                f"ScatterPlot_11Hz_BP_{binStr}_{set_name}_{eo_str}_{scale_str}.png",
            )
        )

    if (binWidth == 2 and not bin2_startF) or binWidth == 1:
        if op_SAGES:
            ttst_O1 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O1_12],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_12],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_12],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_12],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_12],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_12],
                equal_var=False,
            )
            ttst_Oz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_12],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_12],
                equal_var=False,
            )
            ttst_PO3 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_12],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_12],
                equal_var=False,
            )
            ttst_PO4 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_12],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_12],
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
                feats_DF_CAM_Sbj.loc[:, id_O1_12],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_12],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_12],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_12],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_12],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_12],
                equal_var=False,
            )

            pval_list = [
                ttst_O1.pvalue,
                ttst_O2.pvalue,
                ttst_POz.pvalue,
            ]

        x_ticks_lbls = [
            chn_name + " (" + f"{pval:.2f})"
            for (chn_name, pval) in zip(sel_ch_names_12Hz, pval_list)
        ]

        fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        fig.suptitle(
            rf"12 Hz ({binStr} Bin-Avg.) PSD {title_name} {eo_str} {scale_str}"
        )

        plt.plot(
            -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O1_12],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_12],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_12],
            equal_var=False,
        )
        # plt.text(0,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O2_12],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_12],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_12],
            equal_var=False,
        )
        # plt.text(1,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_POz_12],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_12],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_12],
            equal_var=False,
        )
        # plt.text(2,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            -0.2,
            feats_DF_Ctrl_Grp.loc[0, id_O1_12],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            0.8,
            feats_DF_Ctrl_Grp.loc[0, id_O2_12],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.8,
            feats_DF_Ctrl_Grp.loc[0, id_POz_12],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O1_12],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O2_12],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_POz_12],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            0.2,
            feats_DF_CAM_Grp.loc[0, id_O1_12],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.2,
            feats_DF_CAM_Grp.loc[0, id_O2_12],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            2.2,
            feats_DF_CAM_Grp.loc[0, id_POz_12],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        if op_SAGES:
            plt.plot(
                2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_12],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_12],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_12],
                equal_var=False,
            )
            # plt.text(3,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_12],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_12],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_12],
                equal_var=False,
            )
            # plt.text(4,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_12],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_12],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_12],
                equal_var=False,
            )
            # plt.text(5,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                2.8,
                feats_DF_Ctrl_Grp.loc[0, id_Oz_12],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO3_12],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO4_12],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_Oz_12],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO3_12],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO4_12],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                3.2,
                feats_DF_CAM_Grp.loc[0, id_Oz_12],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                4.2,
                feats_DF_CAM_Grp.loc[0, id_PO3_12],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                5.2,
                feats_DF_CAM_Grp.loc[0, id_PO4_12],
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
                figDir,
                f"ScatterPlot_12Hz_BP_{binStr}_{set_name}_{eo_str}_{scale_str}.png",
            )
        )

    if (binWidth == 2 and not bin2_startF) or binWidth == 1:
        if op_SAGES:
            ttst_O1 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O1_4],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_4],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_4],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_4],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_4],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_4],
                equal_var=False,
            )
            ttst_Oz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_4],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_4],
                equal_var=False,
            )
            ttst_PO3 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_4],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_4],
                equal_var=False,
            )
            ttst_PO4 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_4],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_4],
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
                feats_DF_CAM_Sbj.loc[:, id_O1_4],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_4],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_4],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_4],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_4],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_4],
                equal_var=False,
            )

            pval_list = [
                ttst_O1.pvalue,
                ttst_O2.pvalue,
                ttst_POz.pvalue,
            ]

        x_ticks_lbls = [
            chn_name + " (" + f"{pval:.2f})"
            for (chn_name, pval) in zip(sel_ch_names_4Hz, pval_list)
        ]

        fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        fig.suptitle(rf"4 Hz ({binStr} Bin-Avg.) PSD {title_name} {eo_str} {scale_str}")

        plt.plot(
            -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O1_4],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_4],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_4],
            equal_var=False,
        )
        # plt.text(0,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O2_4],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_4],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_4],
            equal_var=False,
        )
        # plt.text(1,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_POz_4],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_4],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_4],
            equal_var=False,
        )
        # plt.text(2,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            -0.2,
            feats_DF_Ctrl_Grp.loc[0, id_O1_4],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            0.8,
            feats_DF_Ctrl_Grp.loc[0, id_O2_4],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.8,
            feats_DF_Ctrl_Grp.loc[0, id_POz_4],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O1_4],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O2_4],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_POz_4],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            0.2,
            feats_DF_CAM_Grp.loc[0, id_O1_4],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.2,
            feats_DF_CAM_Grp.loc[0, id_O2_4],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            2.2,
            feats_DF_CAM_Grp.loc[0, id_POz_4],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        if op_SAGES:
            plt.plot(
                2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_4],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_4],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_4],
                equal_var=False,
            )
            # plt.text(3,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_4],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_4],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_4],
                equal_var=False,
            )
            # plt.text(4,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_4],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_4],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_4],
                equal_var=False,
            )
            # plt.text(5,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                2.8,
                feats_DF_Ctrl_Grp.loc[0, id_Oz_4],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO3_4],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO4_4],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_Oz_4],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO3_4],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO4_4],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                3.2,
                feats_DF_CAM_Grp.loc[0, id_Oz_4],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                4.2,
                feats_DF_CAM_Grp.loc[0, id_PO3_4],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                5.2,
                feats_DF_CAM_Grp.loc[0, id_PO4_4],
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
                figDir,
                f"ScatterPlot_4Hz_BP_{binStr}_{set_name}_{eo_str}_{scale_str}.png",
            )
        )

    if (binWidth == 2 and bin2_startF) or binWidth == 1:
        if op_SAGES:
            ttst_O1 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O1_5],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_5],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_5],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_5],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_5],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_5],
                equal_var=False,
            )
            ttst_Oz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_5],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_5],
                equal_var=False,
            )
            ttst_PO3 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_5],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_5],
                equal_var=False,
            )
            ttst_PO4 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_5],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_5],
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
                feats_DF_CAM_Sbj.loc[:, id_O1_5],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_5],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_5],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_5],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_5],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_5],
                equal_var=False,
            )

            pval_list = [
                ttst_O1.pvalue,
                ttst_O2.pvalue,
                ttst_POz.pvalue,
            ]

        x_ticks_lbls = [
            chn_name + " (" + f"{pval:.2f})"
            for (chn_name, pval) in zip(sel_ch_names_5Hz, pval_list)
        ]

        fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        fig.suptitle(rf"5 Hz ({binStr} Bin-Avg.) PSD {title_name} {eo_str} {scale_str}")

        plt.plot(
            -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O1_5],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_5],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_5],
            equal_var=False,
        )
        # plt.text(0,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O2_5],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_5],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_5],
            equal_var=False,
        )
        # plt.text(1,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_POz_5],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_5],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_5],
            equal_var=False,
        )
        # plt.text(2,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            -0.2,
            feats_DF_Ctrl_Grp.loc[0, id_O1_5],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            0.8,
            feats_DF_Ctrl_Grp.loc[0, id_O2_5],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.8,
            feats_DF_Ctrl_Grp.loc[0, id_POz_5],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O1_5],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O2_5],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_POz_5],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            0.2,
            feats_DF_CAM_Grp.loc[0, id_O1_5],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.2,
            feats_DF_CAM_Grp.loc[0, id_O2_5],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            2.2,
            feats_DF_CAM_Grp.loc[0, id_POz_5],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        if op_SAGES:
            plt.plot(
                2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_5],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_5],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_5],
                equal_var=False,
            )
            # plt.text(3,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_5],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_5],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_5],
                equal_var=False,
            )
            # plt.text(4,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_5],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_5],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_5],
                equal_var=False,
            )
            # plt.text(5,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                2.8,
                feats_DF_Ctrl_Grp.loc[0, id_Oz_5],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO3_5],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO4_5],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_Oz_5],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO3_5],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO4_5],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                3.2,
                feats_DF_CAM_Grp.loc[0, id_Oz_5],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                4.2,
                feats_DF_CAM_Grp.loc[0, id_PO3_5],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                5.2,
                feats_DF_CAM_Grp.loc[0, id_PO4_5],
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
                figDir,
                f"ScatterPlot_5Hz_BP_{binStr}_{set_name}_{eo_str}_{scale_str}.png",
            )
        )

    if (binWidth == 2 and not bin2_startF) or binWidth == 1:
        if op_SAGES:
            ttst_O1 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O1_6],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_6],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_6],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_6],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_6],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_6],
                equal_var=False,
            )
            ttst_Oz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_6],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_6],
                equal_var=False,
            )
            ttst_PO3 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_6],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_6],
                equal_var=False,
            )
            ttst_PO4 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_6],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_6],
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
                feats_DF_CAM_Sbj.loc[:, id_O1_6],
                feats_DF_Ctrl_Sbj.loc[:, id_O1_6],
                equal_var=False,
            )
            ttst_O2 = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_O2_6],
                feats_DF_Ctrl_Sbj.loc[:, id_O2_6],
                equal_var=False,
            )
            ttst_POz = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_POz_6],
                feats_DF_Ctrl_Sbj.loc[:, id_POz_6],
                equal_var=False,
            )

            pval_list = [
                ttst_O1.pvalue,
                ttst_O2.pvalue,
                ttst_POz.pvalue,
            ]

        x_ticks_lbls = [
            chn_name + " (" + f"{pval:.2f})"
            for (chn_name, pval) in zip(sel_ch_names_6Hz, pval_list)
        ]

        fig, axs = plt.subplots(figsize=(7.5, 4.5), nrows=1, ncols=1)
        fig.suptitle(rf"6 Hz ({binStr} Bin-Avg.) PSD {title_name} {eo_str} {scale_str}")

        plt.plot(
            -0.2 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O1_6],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O1_6],
            feats_DF_Ctrl_Sbj.loc[:, id_O1_6],
            equal_var=False,
        )
        # plt.text(0,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            0.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_O2_6],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_O2_6],
            feats_DF_Ctrl_Sbj.loc[:, id_O2_6],
            equal_var=False,
        )
        # plt.text(1,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            1.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
            feats_DF_Ctrl_Sbj.loc[:, id_POz_6],
            marker="o",
            color="b",
            linestyle="None",
        )
        tmp_pval = scipy.stats.ttest_ind(
            feats_DF_CAM_Sbj.loc[:, id_POz_6],
            feats_DF_Ctrl_Sbj.loc[:, id_POz_6],
            equal_var=False,
        )
        # plt.text(2,1.2,f"{tmp_pval.pvalue:.2f}")

        plt.plot(
            -0.2,
            feats_DF_Ctrl_Grp.loc[0, id_O1_6],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            0.8,
            feats_DF_Ctrl_Grp.loc[0, id_O2_6],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.8,
            feats_DF_Ctrl_Grp.loc[0, id_POz_6],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        plt.plot(
            0.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O1_6],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            1.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_O2_6],
            marker="o",
            color="m",
            linestyle="None",
        )
        plt.plot(
            2.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
            feats_DF_CAM_Sbj.loc[:, id_POz_6],
            marker="o",
            color="m",
            linestyle="None",
        )

        plt.plot(
            0.2,
            feats_DF_CAM_Grp.loc[0, id_O1_6],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            1.2,
            feats_DF_CAM_Grp.loc[0, id_O2_6],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )
        plt.plot(
            2.2,
            feats_DF_CAM_Grp.loc[0, id_POz_6],
            marker="s",
            markersize=10,
            color="k",
            linestyle="None",
        )

        if op_SAGES:
            plt.plot(
                2.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_6],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_Oz_6],
                feats_DF_Ctrl_Sbj.loc[:, id_Oz_6],
                equal_var=False,
            )
            # plt.text(3,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                3.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_6],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO3_6],
                feats_DF_Ctrl_Sbj.loc[:, id_PO3_6],
                equal_var=False,
            )
            # plt.text(4,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                4.8 * np.ones(feats_DF_Ctrl_Sbj.shape[0]),
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_6],
                marker="o",
                color="b",
                linestyle="None",
            )
            tmp_pval = scipy.stats.ttest_ind(
                feats_DF_CAM_Sbj.loc[:, id_PO4_6],
                feats_DF_Ctrl_Sbj.loc[:, id_PO4_6],
                equal_var=False,
            )
            # plt.text(5,1.2,f"{tmp_pval.pvalue:.2f}")

            plt.plot(
                2.8,
                feats_DF_Ctrl_Grp.loc[0, id_Oz_6],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO3_6],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                4.8,
                feats_DF_Ctrl_Grp.loc[0, id_PO4_6],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                3.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_Oz_6],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                4.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO3_6],
                marker="o",
                color="m",
                linestyle="None",
            )
            plt.plot(
                5.2 * np.ones(feats_DF_CAM_Sbj.shape[0]),
                feats_DF_CAM_Sbj.loc[:, id_PO4_6],
                marker="o",
                color="m",
                linestyle="None",
            )

            plt.plot(
                3.2,
                feats_DF_CAM_Grp.loc[0, id_Oz_6],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )

            plt.plot(
                4.2,
                feats_DF_CAM_Grp.loc[0, id_PO3_6],
                marker="s",
                markersize=10,
                color="k",
                linestyle="None",
            )
            plt.plot(
                5.2,
                feats_DF_CAM_Grp.loc[0, id_PO4_6],
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
                figDir,
                f"ScatterPlot_6Hz_BP_{binStr}_{set_name}_{eo_str}_{scale_str}.png",
            )
        )

        plt.close("all")
