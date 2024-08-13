"""
Created on Jan 24, 2024

@author: mning
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


def f1_custom(y_true, y_pred):
    rcll_scr = recall_score(y_true, y_pred)
    prc_scr = precision_score(y_true, y_pred)

    return 2 / (rcll_scr ** (-1) + prc_scr ** (-1))
