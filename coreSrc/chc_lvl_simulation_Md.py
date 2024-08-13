"""
Created on Jan 19, 2024

@author: mning
"""
from dataclasses import dataclass
from math import ceil

import numpy as np
from numpy.random import uniform
from sklearn import metrics


@dataclass
class ScoresTuple:
    acc: float = 0.0
    sen: float = 0.0
    spc: float = 0.0
    f1: float = 0.0
    roc_auc: float = 0.0
    pr_auc: float = 0.0
    ppv: float = 0.0
    npv: float = 0.0


# only for 2 classes for now
def run_Chc_Sim(
    prob_pos,
    prob_neg,
    samp_pos,
    samp_neg,
    num_rep=100000,
) -> ScoresTuple:
    # prob_pos = 6/51
    # prob_neg = 45/51
    # samp_pos = 6/51
    # samp_neg = 45/51

    rnd_nmb_inst = uniform(low=0.0, high=1.0, size=num_rep)
    # for true labels: set samp_neg to class 0 and samp_pos to class 1
    # for predicted labels: if below prob_neg, assigned class 0. else 1
    true_lbls = np.zeros((1, num_rep))
    true_lbls[0, ceil(num_rep * samp_neg) :] = 1
    true_lbls = true_lbls.reshape(-1)
    prd_lbls = np.asarray([0 if el < prob_neg else 1 for el in rnd_nmb_inst])

    # get scores
    acc = metrics.accuracy_score(true_lbls, prd_lbls)
    sen = metrics.recall_score(true_lbls, prd_lbls)
    spc = metrics.recall_score(true_lbls, prd_lbls, pos_label=0)
    f1 = metrics.f1_score(true_lbls, prd_lbls)
    roc_auc = metrics.roc_auc_score(true_lbls, prd_lbls)
    pr_auc = metrics.average_precision_score(true_lbls, prd_lbls)
    prc = metrics.precision_score(true_lbls, prd_lbls)

    return ScoresTuple(acc, sen, spc, f1, roc_auc, pr_auc, prc)


def get_Theor_Chc(
    prob_pos: float,
    prob_neg: float,
    samp_pos: float,
    samp_neg: float,
) -> ScoresTuple:
    chc_acc = prob_pos * samp_pos + prob_neg * samp_neg
    chc_sen = (prob_pos * samp_pos) / (prob_pos * samp_pos + prob_neg * samp_pos)
    chc_spc = (prob_neg * samp_neg) / (prob_pos * samp_neg + prob_neg * samp_neg)
    chc_roc_auc = 0.5
    chc_f1 = (2 * prob_pos * samp_pos) / (prob_pos + samp_pos)
    chc_pr_auc = samp_pos
    chc_ppv = (prob_pos * samp_pos) / (prob_pos * samp_pos + prob_pos * samp_neg)
    chc_npv = (prob_neg * samp_neg) / (prob_neg * samp_pos + prob_neg * samp_neg)

    return ScoresTuple(
        chc_acc, chc_sen, chc_spc, chc_f1, chc_roc_auc, chc_pr_auc, chc_ppv, chc_npv
    )
