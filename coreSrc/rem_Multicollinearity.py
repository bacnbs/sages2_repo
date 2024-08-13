"""
Created on Sep 15, 2023

Code from here: https://stats.stackexchange.com/questions/155028/how-to-systematically-remove-collinear-variables-pandas-columns-in-python

@author: mning
"""
from statsmodels.stats.outliers_influence import variance_inflation_factor


def remove_vif(X, thresh=5.0):
    X = X.assign(const=1)  # faster than add_constant from statsmodels
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [
            variance_inflation_factor(X.iloc[:, variables].values, ix)
            for ix in range(X.iloc[:, variables].shape[1])
        ]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print(
                "dropping '"
                + X.iloc[:, variables].columns[maxloc]
                + "' at index: "
                + str(maxloc)
            )
            del variables[maxloc]
            dropped = True

    print("Remaining variables:")
    print(X.columns[variables[:-1]])
    # check variable type, i think dataframe
    return X.iloc[:, variables[:-1]]
