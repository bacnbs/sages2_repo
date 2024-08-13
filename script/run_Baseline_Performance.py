"""
Created on Jun 20, 2023

@author: mning

Baseline: MoCA or GCP
MoCA were performed at the end of the day and
is the last exam of a battery of tests.
So MoCA needed to be corrected.

Can try GCP instead.

"""
import itertools
import os
from typing import Any
from typing import Dict

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from EEGFeaturesExtraction.coreSrc.features import getDeliriumStatusMd
from EEGFeaturesExtraction.coreSrc.features import getGCPMd
from EEGFeaturesExtraction.coreSrc.features import getMoCAMD
from EEGFeaturesExtraction.coreSrc.features.getMoCAMD import getMMSE
from EEGFeaturesExtraction.coreSrc.getModelFnMd import getModelFn
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from scipy.stats import t
from scipy.stats import zscore
from sklearn import metrics
from sklearn.metrics import auc
from sklearn.metrics import make_scorer
from sklearn.metrics import recall_score
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection._validation import _fit_and_score
from sklearn.utils import resample

# from sklearn.model_selection import train_test_split

plt.set_loglevel(level="warning")

specificity = make_scorer(recall_score, pos_label=0)

op_Training = "SAGES"
"""
op_Training = {'SAGES','CV'}
"""
testCase = 0
plotOp = 0

is_EO = 1
op_corr = 1
op_Scl_CogAssmnt = 4
# op_sub
#    0 - use MoCA
#    1 - visuospatial executive
#    2 - attention
#    3 - memory
#    4- sum of visuospatial executive, attention & memory
#    5 - GCP
#    6 - MMSE
op_sub = 0
op_eeg = 1
# op_eeg
#    0 - eeg cohort
#    1 - entire SAGES cohort

num_rep_bs = 2000
op_Oversampling = 0

if not op_sub:
    if op_corr:
        moCA_str = "CorrectedMoCA"
    else:
        moCA_str = "OriginalMoCA"
elif op_sub == 1:
    moCA_str = "VisuoSpa"
elif op_sub == 2:
    moCA_str = "Attention"
elif op_sub == 3:
    moCA_str = "Memory"
elif op_sub == 4:
    moCA_str = "Sum3SubMoCA"
elif op_sub == 5:
    moCA_str = "GCP"
elif op_sub == 6:
    moCA_str = "MMSE"

if op_eeg:
    moCA_str += "_EEG"
else:
    moCA_str += "_Cohort"

figDir = (
    r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data"
    r"\Baseline_rsEEG_preprocessing\DATA_visualization\Group\Performances"
)

csvDir = (
    r"C:\Users\mning\Documents\CNBS_Projects\SAGES I and II\Data"
    r"\Baseline_rsEEG_preprocessing\csv_tests"
)

if not op_eeg:
    if not op_sub:
        SAGES_DF = getMoCAMD.getMoCACohort(op_corr)
    elif op_sub == 1:
        SAGES_DF = getMoCAMD.getMoCACohort(op_corr)
        SAGES_DF = SAGES_DF.with_columns(
            pl.col("vdmoca_visuospatial_executive").alias("MoCA")
        )
    elif op_sub == 2:
        SAGES_DF = getMoCAMD.getMoCACohort(op_corr)
        SAGES_DF = SAGES_DF.with_columns(pl.col("vdmoca_attention").alias("MoCA"))
    elif op_sub == 3:
        SAGES_DF = getMoCAMD.getMoCACohort(op_corr)
        SAGES_DF = SAGES_DF.with_columns(
            pl.col("vdmoca_memory_alternative_items").alias("MoCA")
        )
    elif op_sub == 4:
        SAGES_DF = getMoCAMD.getMoCACohort(op_corr)
        SAGES_DF = SAGES_DF.with_columns(
            (
                pl.col("vdmoca_visuospatial_executive")
                + pl.col("vdmoca_attention")
                + pl.col("vdmoca_memory_alternative_items")
            ).alias("MoCA")
        )
    MMSE_Duke = getMoCAMD.getMMSE()
    MoCA_Duke = getMoCAMD.convert_MMSE_MoCA(MMSE_Duke)
else:
    if not op_sub:
        MoCA_SAGES = getMoCAMD.getMoCA(is_EO, op_corr)
        MMSE_Duke = getMoCAMD.getMMSE()
        MoCA_Duke = getMoCAMD.convert_MMSE_MoCA(MMSE_Duke)
    elif op_sub == 1:
        MoCA_SAGES = getMoCAMD.getSpatialEx()
    elif op_sub == 2:
        MoCA_SAGES = getMoCAMD.getAttention()
    elif op_sub == 3:
        MoCA_SAGES = getMoCAMD.getMemory()
    elif op_sub == 4:
        MoCA_SAGES = getMoCAMD.getMoCA(is_EO, op_corr)
        spa_dict = getMoCAMD.getSpatialEx()
        att_dict = getMoCAMD.getAttention()
        mem_dict = getMoCAMD.getMemory()
        for k in MoCA_SAGES.keys():
            MoCA_SAGES[k] = spa_dict[k] + att_dict[k] + mem_dict[k]
    elif op_sub == 5:
        MoCA_SAGES = getGCPMd.getGCP(is_EO)
    elif op_sub == 6:
        temp_dict = getMoCAMD.getMoCA(is_EO, op_corr)
        temp_dict_round = {k: round(v) for (k, v) in temp_dict.items()}
        MoCA_SAGES = getMoCAMD.convert_MoCA_MMSE(temp_dict_round)
        MoCA_Duke = getMoCAMD.getMMSE()

    k_SAGES, v_SAGES = zip(*MoCA_SAGES.items())
    # SAGES_DF = pd.DataFrame({"# SbjID": k_SAGES, "MoCA": v_SAGES})
    SAGES_DF = pl.from_dicts([k_SAGES, v_SAGES], schema=["# SbjID", "MoCA"])
    # SAGES_DF["MoCA"] = SAGES_DF.loc[:, ["MoCA"]].apply(zscore, nan_policy="omit")

k, v = zip(*MoCA_Duke.items())
# DUKE_DF = pl.from_dicts(MoCA_Duke, schema=["# SbjID", "MoCA"])
DUKE_DF = pl.DataFrame([k, v], schema=["# SbjID", "MoCA"])
# DUKE_DF = pd.DataFrame({"# SbjID": k, "MoCA": v})


# Unlike pandas, polars doesn't allow much mutations,
# so usually create new dataframe instead of modifying existing one
DUKE_DF = DUKE_DF.filter(~pl.col("# SbjID").is_in(["0190", "046"]))

# transformation
mu_SAGES = SAGES_DF["MoCA"].mean()
# ddof = 0
sigma_SAGES = SAGES_DF["MoCA"].std(ddof=0)

if op_Scl_CogAssmnt == 0 or op_Scl_CogAssmnt == 3 or op_Scl_CogAssmnt == 4:
    SAGES_DF = SAGES_DF.with_columns(
        ((SAGES_DF["MoCA"] - mu_SAGES) / sigma_SAGES).alias("MoCA_Tr")
    )
elif op_Scl_CogAssmnt == 2:
    # figure out how to do this in polars
    pass
    # SAGES_DF["MoCA"] = (
    #     SAGES_DF["MoCA"]
    #     .to_frame()
    #     .apply(
    #         lambda x: np.where(
    #             np.abs(np.log10(x) - np.log10(x.median())) != 0,
    #             np.abs(np.log10(x) - np.log10(x.median())),
    #             np.NaN,
    #         )
    #     )
    # )
else:
    pass

if op_Scl_CogAssmnt == 0:
    mu_DUKES = DUKE_DF["MoCA"].mean()
    # ddof = 0
    sigma_DUKES = DUKE_DF["MoCA"].std(ddof=0)

    DUKE_DF = DUKE_DF.with_columns(
        ((DUKE_DF["MoCA"] - mu_DUKES) / sigma_DUKES).alias("MoCA_Tr")
    )
elif op_Scl_CogAssmnt == 4:
    DUKE_DF = DUKE_DF.with_columns(
        ((DUKE_DF["MoCA"] - mu_SAGES) / sigma_SAGES).alias("MoCA_Tr")
    )
elif op_Scl_CogAssmnt == 2:
    pass
    # DUKE_DF["MoCA"] = (
    #     DUKE_DF["MoCA"]
    #     .to_frame()
    #     .apply(
    #         lambda x: np.where(
    #             np.abs(np.log10(x) - np.log10(x.median())) != 0,
    #             np.abs(np.log10(x) - np.log10(x.median())),
    #             np.NaN,
    #         )
    #     )
    # )
# PMCID: PMC5545909
# PMID: 28697562
elif op_Scl_CogAssmnt == 3:
    n_tot_SWEDEN = 758 + 102
    mu_SWEDEN = (26 * 758 + 21.6 * 102) / (758 + 102)
    q1 = (2.3**2) * 758 + 758 * (26**2)
    q2 = (4.3**2) * 102 + 102 * (21.6**2)
    qc = q1 + q2
    std_SWEDEN = ((qc - (758 + 102) * mu_SWEDEN**2) / (758 + 102)) ** (1 / 2)
    DUKE_DF = DUKE_DF.with_columns(
        ((DUKE_DF["MoCA"] - mu_SWEDEN) / std_SWEDEN).alias("MoCA_Tr")
    )
else:
    pass

# responses_DUKE.drop(rows_to_del.index, inplace=True)

CAM = getDeliriumStatusMd.getDeliriumStatus(1)
Duke_Del = getDeliriumStatusMd.getDeliriumStatus(0)

# try {0,1} and {-1,1} encoding. will get different results.
predictors_SAGES_DF = SAGES_DF.select("MoCA_Tr")
if op_eeg:
    responses_SAGES = np.where(SAGES_DF["# SbjID"].is_in(CAM), 1, 0)
    responses_inv_SAGES = np.where(SAGES_DF["# SbjID"].is_in(CAM), 0, 1)
else:
    temp_SAGES = SAGES_DF.with_columns(
        pl.when(pl.col("vdsagesdeliriumever") == 1)
        .then(0)
        .otherwise(1)
        .alias("delirium_inv")
    )
    responses_SAGES = temp_SAGES.select("vdsagesdeliriumever").to_numpy()
    responses_inv_SAGES = temp_SAGES.select("delirium_inv").to_numpy()

predictors_Duke_DF = DUKE_DF.select("MoCA_Tr")
responses_Duke = np.where(DUKE_DF["# SbjID"].is_in(Duke_Del), 1, 0).reshape(-1, 1)
responses_inv_Duke = np.where(DUKE_DF["# SbjID"].is_in(Duke_Del), 0, 1).reshape(-1, 1)

params: Dict[str, Any] = dict()

mdlDict = getModelFn(params)
mdl_Scores: Dict[str, Any] = dict.fromkeys(mdlDict.keys(), 0)
mdl_Acc = np.zeros((len(mdlDict)))
mdl_F1 = np.zeros((len(mdlDict)))
mdl_ROC_AUC = np.zeros((len(mdlDict)))
mdl_sensitivity = np.zeros((len(mdlDict)))
mdl_specificity = np.zeros((len(mdlDict)))
mdl_PR_AUC = np.zeros((len(mdlDict)))
mdl_PPV = np.zeros((len(mdlDict)))
mdl_NPV = np.zeros((len(mdlDict)))
if op_Training == "CV":
    mdl_Acc_CI = np.zeros((len(mdlDict)))
    mdl_Sen_CI = np.zeros((len(mdlDict)))
    mdl_Spc_CI = np.zeros((len(mdlDict)))
    mdl_F1_CI = np.zeros((len(mdlDict)))
    mdl_ROC_AUC_CI = np.zeros((len(mdlDict)))
    mdl_PR_AUC_CI = np.zeros((len(mdlDict)))
    mdl_PPV_CI = np.zeros((len(mdlDict)))
    mdl_NPV_CI = np.zeros((len(mdlDict)))

dof_SAGES = SAGES_DF.shape[0] - 2

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
# stratified cross-validation
# avoid hyperparameter, use cross-validation
# skf = StratifiedKFold(n_splits=5, shuffle=True)
skf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5)

mdl_Acc_bs = np.zeros((len(mdlDict), num_rep_bs))
mdl_F1_bs = np.zeros((len(mdlDict), num_rep_bs))
mdl_ROC_AUC_bs = np.zeros((len(mdlDict), num_rep_bs))
mdl_sensitivity_bs = np.zeros((len(mdlDict), num_rep_bs))
mdl_specificity_bs = np.zeros((len(mdlDict), num_rep_bs))
mdl_PR_AUC_bs = np.zeros((len(mdlDict), num_rep_bs))
mdl_PPV_bs = np.zeros((len(mdlDict), num_rep_bs))
mdl_NPV_bs = np.zeros((len(mdlDict), num_rep_bs))
mdlAccCI_bs = np.zeros((len(mdlDict), num_rep_bs))

for i_mdl, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
    if op_Training == "CV":
        print(mdl_name)
        predictors_Arr_SAGES = predictors_SAGES_DF.to_numpy()
        mdl_Scores[mdl_name] = cross_validate(
            mdl_instance,
            predictors_Arr_SAGES,
            responses_SAGES.ravel(),
            cv=skf,
            scoring=scoring,
            error_score="raise",
            return_estimator=True,
        )
        # unknown mean and std
        temp_scores = mdl_Scores[mdl_name]["test_accuracy"]
        mdl_Acc[i_mdl] = np.mean(temp_scores)
        mdl_F1[i_mdl] = np.mean(mdl_Scores[mdl_name]["test_f1"])
        mdl_ROC_AUC[i_mdl] = np.mean(mdl_Scores[mdl_name]["test_roc_auc"])
        mdl_sensitivity[i_mdl] = np.mean(mdl_Scores[mdl_name]["test_sensitivity"])
        mdl_specificity[i_mdl] = np.mean(mdl_Scores[mdl_name]["test_specificity"])
        mdl_PR_AUC[i_mdl] = np.mean(mdl_Scores[mdl_name]["test_pr_auc"])
        mdl_PPV[i_mdl] = np.mean(mdl_Scores[mdl_name]["test_PPV"])
        mdl_NPV[i_mdl] = np.mean(mdl_Scores[mdl_name]["test_NPV"])

        mdl_Acc_CI[i_mdl] = (
            t.ppf(1 - (1 - 0.95) / 2, dof_SAGES)
            * np.std(temp_scores)
            / np.sqrt(np.shape(temp_scores)[0])
        )

        mdl_Sen_CI[i_mdl] = (
            t.ppf(1 - (1 - 0.95) / 2, dof_SAGES)
            * np.std(mdl_Scores[mdl_name]["test_sensitivity"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_sensitivity"])[0])
        )

        mdl_Spc_CI[i_mdl] = (
            t.ppf(1 - (1 - 0.95) / 2, dof_SAGES)
            * np.std(mdl_Scores[mdl_name]["test_specificity"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_specificity"])[0])
        )

        mdl_F1_CI[i_mdl] = (
            t.ppf(1 - (1 - 0.95) / 2, dof_SAGES)
            * np.std(mdl_Scores[mdl_name]["test_f1"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_f1"])[0])
        )

        mdl_ROC_AUC_CI[i_mdl] = (
            t.ppf(1 - (1 - 0.95) / 2, dof_SAGES)
            * np.std(mdl_Scores[mdl_name]["test_roc_auc"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_roc_auc"])[0])
        )

        mdl_PR_AUC_CI[i_mdl] = (
            t.ppf(1 - (1 - 0.95) / 2, dof_SAGES)
            * np.std(mdl_Scores[mdl_name]["test_pr_auc"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_pr_auc"])[0])
        )

        mdl_PPV_CI[i_mdl] = (
            t.ppf(1 - (1 - 0.95) / 2, dof_SAGES)
            * np.std(mdl_Scores[mdl_name]["test_PPV"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_PPV"])[0])
        )

        mdl_NPV_CI[i_mdl] = (
            t.ppf(1 - (1 - 0.95) / 2, dof_SAGES)
            * np.std(mdl_Scores[mdl_name]["test_NPV"])
            / np.sqrt(np.shape(mdl_Scores[mdl_name]["test_NPV"])[0])
        )

    elif op_Training == "SAGES":
        predictors_Arr_SAGES = predictors_SAGES_DF.to_numpy()
        predictors_Arr_Duke = predictors_Duke_DF.to_numpy()

        # test
        # sages_csv = os.path.join(csvDir, "sages_pred.csv")
        # predictors_SAGES_DF.write_csv(sages_csv)

        predictors_Arr = np.concatenate(
            (predictors_Arr_SAGES, predictors_Arr_Duke), axis=0
        )
        responses_SAGES = responses_SAGES.reshape(-1, 1)
        responses = np.concatenate((responses_SAGES, responses_Duke), axis=0).ravel()

        train_idx = [i for i in range(predictors_Arr_SAGES.shape[0])]
        test_idx = [
            i
            for i in range(
                predictors_Arr_SAGES.shape[0],
                predictors_Arr_SAGES.shape[0] + predictors_Arr_Duke.shape[0],
            )
        ]

        print(mdl_name)

        # Bootstrapping

        ## Actually not needed.
        dof_bs = num_rep_bs - 1

        ## Perform over-sampling on the minority class before bootstrapping
        if op_Oversampling == 1:
            ros = RandomOverSampler(random_state=0, sampling_strategy=0.6, shrinkage=1)
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

        mdl_instance.fit(predictors_Arr_SAGES_os, responses_SAGES_os.ravel())
        mdl_Scores[mdl_name] = mdl_instance
        for iter_bs in range(num_rep_bs):
            predictors_DUKE_bs, responses_DUKE_bs = resample(
                predictors_Arr_Duke,
                responses_Duke,
                n_samples=predictors_Arr_Duke.shape[0],
                stratify=responses_Duke,
            )

            responses_DUKE_bs_pred = mdl_instance.predict(predictors_DUKE_bs)
            if mdl_name == "NSC Custom EC" or mdl_name == "NSC MH":
                responses_DUKE_bs_dec_fnc = mdl_instance.predict(predictors_DUKE_bs)
            elif mdl_name == "GNB" or mdl_name == "GNB Prior" or mdl_name == "Tree":
                responses_DUKE_bs_dec_fnc = mdl_instance.predict_proba(
                    predictors_DUKE_bs
                )[:, 1]
            else:
                responses_DUKE_bs_dec_fnc = mdl_instance.decision_function(
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

        responses_DUKE_pred = mdl_instance.predict(predictors_Arr_Duke)
        if mdl_name == "NSC Custom EC" or mdl_name == "NSC MH":
            responses_DUKE_dec_fnc = mdl_instance.predict(predictors_Arr_Duke)
        elif mdl_name == "GNB" or mdl_name == "GNB Prior" or mdl_name == "Tree":
            responses_DUKE_dec_fnc = mdl_instance.predict_proba(predictors_Arr_Duke)[
                :, 1
            ]
        else:
            responses_DUKE_dec_fnc = mdl_instance.decision_function(predictors_Arr_Duke)

        ## Save scores to variables.
        mdl_Acc[i_mdl] = metrics.accuracy_score(responses_Duke, responses_DUKE_pred)
        mdl_ROC_AUC[i_mdl] = metrics.roc_auc_score(
            responses_Duke, responses_DUKE_dec_fnc
        )
        mdl_PR_AUC[i_mdl] = metrics.average_precision_score(
            responses_Duke, responses_DUKE_dec_fnc
        )
        mdl_PPV[i_mdl] = metrics.precision_score(responses_Duke, responses_DUKE_pred)
        mdl_NPV[i_mdl] = metrics.precision_score(
            responses_Duke, responses_DUKE_pred, pos_label=0
        )
        mdl_sensitivity[i_mdl] = metrics.recall_score(
            responses_Duke, responses_DUKE_pred
        )
        mdl_specificity[i_mdl] = metrics.recall_score(
            responses_Duke, responses_DUKE_pred, pos_label=0
        )
        mdl_F1[i_mdl] = 2 / (mdl_PPV[i_mdl] ** (-1) + mdl_sensitivity[i_mdl] ** (-1))

if op_Training == "SAGES":
    mdl_Acc_CI = np.percentile(mdl_Acc_bs, q=[2.5, 97.5], axis=1)
    mdl_F1_CI = np.percentile(mdl_F1_bs, q=[2.5, 97.5], axis=1)
    mdl_ROC_AUC_CI = np.percentile(mdl_ROC_AUC_bs, q=[2.5, 97.5], axis=1)
    mdl_PR_AUC_CI = np.percentile(mdl_PR_AUC_bs, q=[2.5, 97.5], axis=1)
    mdl_PPV_CI = np.percentile(mdl_PPV_bs, q=[2.5, 97.5], axis=1)
    mdl_NPV_CI = np.percentile(mdl_NPV_bs, q=[2.5, 97.5], axis=1)
    mdl_sensitivity_CI = np.percentile(mdl_sensitivity_bs, q=[2.5, 97.5], axis=1)
    mdl_specificity_CI = np.percentile(mdl_specificity_bs, q=[2.5, 97.5], axis=1)

mean_fpr = np.linspace(0, 1, 100)

# plot roc-curve
# you may need to re-run cross-validation with cross_val_predict function
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

# Also add roc curve using threshold for MoCA
predictors_Arr_SAGES = predictors_SAGES_DF.to_numpy()
predictors_Arr_Duke = predictors_Duke_DF.to_numpy()
for i, (mdl_name, mdl_instance) in enumerate(itertools.islice(mdlDict.items(), 0, 2)):
    fig, ax = plt.subplots(figsize=(6, 6))
    for fold, (train, test) in enumerate(
        skf.split(predictors_SAGES_DF, responses_SAGES)
    ):
        tmp_train_pred = predictors_Arr_SAGES[train, :]
        tmp_train_resp = responses_SAGES[train].ravel()

        mdl_instance.fit(tmp_train_pred, tmp_train_resp)
        viz = RocCurveDisplay.from_estimator(
            mdl_instance,
            predictors_Arr_SAGES[test, :],
            responses_SAGES[test],
            name=f"ROC fold {fold}",
            alpha=0.3,
            lw=1,
            ax=ax,
        )
        interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(viz.roc_auc)

    ax.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability: {mdl_name}')",
    )
    ax.axis("square")
    # ax.legend(loc="lower right")
    ax.get_legend().remove()

    plt.savefig(os.path.join(figDir, f"ROC_Baseline_{mdl_name}.png"))

# err_df = pl.DataFrame(
#     {
#         "Accuracy": mdl_Acc_CI,
#         "Sensitivity": mdl_Sen_CI,
#         "Specificity": mdl_Spc_CI,
#         "f1": mdl_F1_CI,
#         "roc_auc": mdl_ROC_AUC_CI,
#         "pr_auc": mdl_PR_AUC_CI,
#         "precision": mdl_PPV_CI,
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

scores_df = pl.DataFrame(
    {
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

perf_csv_fn = os.path.join(figDir, f"Performance_Baseline_csv_{op_Training}.csv")
with open(perf_csv_fn, "w") as f_perf:
    f_perf.write("Model Name,")
    for metric_name in scoring.keys():
        f_perf.write(f"{metric_name} mean, {metric_name} 2.5th, {metric_name} 97.5th,")
    f_perf.write("\n")
    if op_Training == "SAGES":
        for i_row, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
            f_perf.write(f"{mdl_name},")
            for i_col, col_name in enumerate(scores_df[:, :].columns):
                f_perf.write(
                    f"{scores_df[i_row, i_col]},"
                    f"{scores_df[i_row, i_col] - err_df[i_col, 0, i_row]},"
                    f"{scores_df[i_row, i_col] + err_df[i_col, 1, i_row]},"
                )
            f_perf.write("\n")
    else:
        for i_row, (mdl_name, mdl_instance) in enumerate(mdlDict.items()):
            f_perf.write(f"{mdl_name},")
            for i_col, col_name in enumerate(scores_df.iloc[:, 1:].columns):
                f_perf.write(
                    f"{scores_df[i_row, i_col]},"
                    f"{scores_df[i_row, i_col] - err_df[i_row, i_col]},"
                    f"{scores_df[i_row, i_col] + err_df[i_row, i_col]},"
                )
            f_perf.write("\n")

# scores_df.plot.bar()
# scores_df.plot.bar() * err_df.plot.errorbars()

fig, ax = plt.subplots(layout="constrained", figsize=(15, 12))

width = 1 / 8
multiplier = 0

for mdl_row in scores_df.rows(named=True):
    offset = multiplier - (3 / 8)
    rects = ax.bar(offset, mdl_row["Accuracy"], width, color="#1f77b4")
    ax.bar_label(rects, padding=3, fmt="%0.2f")

    offset = multiplier - (2 / 8)
    rects = ax.bar(offset, mdl_row["Sensitivity"], width, color="#ff7f0e")
    ax.bar_label(rects, padding=3, fmt="%0.2f")

    offset = multiplier - (1 / 8)
    rects = ax.bar(offset, mdl_row["Specificity"], width, color="#2ca02c")
    ax.bar_label(rects, padding=3, fmt="%0.2f")

    offset = multiplier
    rects = ax.bar(offset, mdl_row["f1"], width, color="#d62728")
    ax.bar_label(rects, padding=3, fmt="%0.2f")

    offset = multiplier + (1 / 8)
    rects = ax.bar(offset, mdl_row["roc_auc"], width, color="#9467bd")
    ax.bar_label(rects, padding=3, fmt="%0.2f")

    offset = multiplier + (2 / 8)
    rects = ax.bar(offset, mdl_row["pr_auc"], width, color="#8c564b")
    ax.bar_label(rects, padding=3, fmt="%0.2f")

    offset = multiplier + (3 / 8)
    rects = ax.bar(offset, mdl_row["PPV"], width, color="#e377c2")
    ax.bar_label(rects, padding=3, fmt="%0.2f")

    offset = multiplier + (4 / 8)
    rects = ax.bar(offset, mdl_row["NPV"], width, color="#F9F871")
    ax.bar_label(rects, padding=3, fmt="%0.2f")

    multiplier += 1

ax.set_ylabel("Length (mm)")
ax.set_title(f"Baseline Performance: {moCA_str}. Training Set: {op_Training}")
ax.set_xticks(ticks=range(0, len(mdlDict.keys())), labels=mdlDict.keys(), rotation=45)
ax.legend(loc="upper right", labels=scoring.keys())
ax.set_ylim(0, 1.2)

plt.savefig(os.path.join(figDir, f"Performance_Baseline_{moCA_str}_{op_Training}.png"))

# save to csv

# roc_curve
# make sure responses_SAGES and v are correctly aligned
if op_Training == "CV":
    fpr, tpr, thresholds = metrics.roc_curve(
        responses_SAGES, predictors_Arr_SAGES, pos_label=0
    )
    auc_roc = metrics.roc_auc_score(
        responses_inv_SAGES, predictors_Arr_SAGES, average="weighted"
    )
    # specificity = metrics.recall_score(responses_inv_SAGES, v)
    # sensitivity = metrics.recall_score(responses_SAGES, v)

    plt.subplots(layout="constrained", figsize=(15, 12))
    plt.plot(fpr, tpr)
    plt.xlabel("false positive rate")
    plt.ylabel("true positive rate")
    # plt.title(f'Area under curve {auc_roc:.2f}. Sensitivity={sensitivity:.2f}. Specificity={specificity:.2f}')
    plt.title(f"Area under curve {auc_roc:.2f} {moCA_str}. {op_Training}")
    plt.savefig(
        os.path.join(
            figDir, f"Performance_Baseline_ROCAUC_{moCA_str}_{op_Training}.png"
        )
    )

    if not op_sub:
        # idk if it's < or <=
        threshold_id8 = [
            1 if this_v <= thresholds[8] else 0 for this_v in predictors_Arr_SAGES
        ]
        threshold_id9 = [
            1 if this_v <= thresholds[9] else 0 for this_v in predictors_Arr_SAGES
        ]

        tn_id8, fp_id8, fn_id8, tp_id8 = metrics.confusion_matrix(
            responses_SAGES, threshold_id8
        ).ravel()
        tn_id9, fp_id9, fn_id9, tp_id9 = metrics.confusion_matrix(
            responses_SAGES, threshold_id9
        ).ravel()

        # sensitivity and tpr are the same
        tpr_id8 = tp_id8 / (tp_id8 + fn_id8)
        fpr_id8 = fp_id8 / (fp_id8 + tn_id8)
        specificity_id8 = tn_id8 / (tn_id8 + fp_id8)

        print(
            f"At index 8, sensitivity: {tpr_id8:.2f} and specificity: {specificity_id8:.2f}"
        )

        tpr_id9 = tp_id9 / (tp_id9 + fn_id9)
        fpr_id9 = fp_id9 / (fp_id9 + tn_id9)
        specificity_id9 = tn_id9 / (tn_id9 + fp_id9)

        print(
            f"At index 9, sensitivity: {tpr_id9:.2f} and specificity: {specificity_id9:.2f}"
        )
