"""
Created on May 22, 2023

@author: mning
"""
"""
Created on May 10, 2023

@author: mning

Note: all must accept same input arguments and yield same output types
"""
import logging

import numpy as np
from EEGFeaturesExtraction.coreSrc.estimators import _NSC
from sklearn.covariance import LedoitWolf
from sklearn.covariance import OAS
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import (
    LogisticRegressionCV,
)
from sklearn.metrics import make_scorer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import (
    NearestCentroid,
)
from sklearn.tree import DecisionTreeClassifier

# import getMin
# import getSTD
logger = logging.getLogger("root")

modelDict = dict


def getModelFn(params: dict) -> modelDict:
    # params contain data-derived parameters

    if "groups" in params:
        params["groups"]

    modelList = {}
    # https://stats.stackexchange.com/questions/48786/algebra-of-lda-fisher-discrimination-power-of-a-variable-and-linear-discriminan/48859#48859
    # https://stats.stackexchange.com/questions/109747/why-is-pythons-scikit-learn-lda-not-working-correctly-and-how-does-it-compute-l/110697#110697
    # modelList.update({"LDA": LinearDiscriminantAnalysis(solver='svd',
    #                                                    shrinkage=None,
    #                                                    priors=[0.5,0.5])})
    #

    modelList.update(
        {
            "LDA CV": GridSearchCV(
                LinearDiscriminantAnalysis(solver="lsqr", priors=[0.75, 0.25]),
                param_grid={"shrinkage": np.arange(0, 1, 0.05)},
                scoring="f1",
                cv=3,
            )
        }
    )

    LW = LedoitWolf(store_precision=False, assume_centered=False)
    modelList.update(
        {
            "LDA LW": LinearDiscriminantAnalysis(
                solver="lsqr", covariance_estimator=LW, priors=[0.75, 0.25]
            )
        }
    )

    # replace this with graphical_lasso
    # gl = GraphicalLasso(alpha=0.2, mode='lars', max_iter = 1000)
    # modelList.update({"LDA GL": LinearDiscriminantAnalysis(solver='lsqr',
    #                                               covariance_estimator=gl,
    #                                               priors=[0.5,0.5])})
    # add priors as [0.5, 0.5]
    oa = OAS(store_precision=False, assume_centered=False)
    modelList.update(
        {
            "LDA OA": LinearDiscriminantAnalysis(
                solver="lsqr", covariance_estimator=oa, priors=[0.75, 0.25]
            )
        }
    )

    # remove QDA, never work well
    # modelList.update({"QDA": QuadraticDiscriminantAnalysis()})

    # modelList.update({"Lgstic Rgrssn L2": LogisticRegression(penalty='l2', solver='lbfgs')})
    #
    # modelList.update({"Lgstic Rgrssn L1": LogisticRegression(penalty='l1', solver='liblinear')})
    #
    # modelList.update({"Lgstic Rgrssn EN": LogisticRegression(penalty='elasticnet',
    #                                                          solver='saga',
    #                                                          l1_ratio=0.02,
    #                                                          max_iter=1500)})
    #
    # modelList.update({"GaussianNB": GaussianNB()})
    #
    modelList.update(
        {"NSC MH": NearestCentroid(metric="manhattan", shrink_threshold=0.2)}
    )
    modelList.update(
        {
            "NSC Custom EC": _NSC.NearestCentroid(
                metric="euclidean", shrink_threshold=0.2
            )
        }
    )
    #
    # modelList.update({"Lgstic Rgrssn GL": LogisticGroupLasso(
    #     groups=groups,
    #     group_reg=0.05,
    #     l1_reg=0.05,
    #     scale_reg="inverse_group_size",
    #     supress_warning=True,
    #     subsampling_scheme=0.1,
    #     n_iter=1500
    #     )})

    # TODO: add ths: https://github.com/glm-tools/pyglmnet
    # binomial distribution with logit link function for generalized linear model

    # compare different solvers for logistic regression

    # gl = GroupLasso(
    #     groups=groups,
    #     group_reg=5,
    #     l1_reg=0,
    #     frobenius_lipschitz=True,
    #     scale_reg="inverse_group_size",
    #     subsampling_scheme=1,
    #     supress_warning=True,
    #     n_iter=1000,
    #     tol=1e-3,
    # )

    return modelList


def getModelNSCFn() -> modelDict:
    modelList = {}
    modelList.update(
        {
            "NSC MH": GridSearchCV(
                NearestCentroid(metric="manhattan"),
                param_grid={"shrink_threshold": np.arange(0.05, 1, 0.05)},
                cv=3,
            )
        }
    )
    return modelList


def getModelCVFn() -> modelDict:
    # if 'groups' in params:
    #     groups = params['groups']

    modelList = {}

    # skf = RepeatedStratifiedKFold(n_splits=3, n_repeats=2)

    # lw = LedoitWolf(store_precision=False, assume_centered=False)
    # oa = OAS(store_precision=False, assume_centered=False)

    # modelList.update({"Lasso Linear CV": LassoCV()})
    # modelList.update({"Lasso L1 CV": LogisticRegressionCV(penalty='l1',
    #                                                       solver='liblinear',
    #                                                       scoring=make_scorer(roc_auc_score),
    #                                                       cv=3,
    #                                                       max_iter = 500)})
    #
    modelList.update(
        {
            "Lasso L2 CV": LogisticRegressionCV(
                penalty="l2", scoring=make_scorer(roc_auc_score), cv=3, max_iter=500
            )
        }
    )
    #
    # modelList.update({"Lasso EN CV": LogisticRegressionCV(penalty='elasticnet',
    #                                                       solver='saga',
    #                                                       l1_ratios=np.arange(0.05,1,0.05),
    #                                                       scoring=make_scorer(roc_auc_score),
    #                                                       cv=3,
    #                                                       max_iter = 500)})

    # modelList.update({"Lgstic Rgrssn L2": LogisticRegression(penalty='l2',
    #                                                          solver='lbfgs',
    #                                                          C=0.001)})

    # modelList.update({"Lgstic Rgrssn L1": LogisticRegression(penalty='l1',
    #                                                          solver='liblinear',
    #                                                          C=0.5)})

    modelList.update(
        {
            "LDA CV": GridSearchCV(
                LinearDiscriminantAnalysis(solver="lsqr", priors=[0.75, 0.25]),
                param_grid={"shrinkage": np.arange(0, 1, 0.05)},
                scoring="roc_auc",
                cv=3,
            )
        }
    )

    LW = LedoitWolf(store_precision=False, assume_centered=False)
    modelList.update(
        {
            "LDA LW": LinearDiscriminantAnalysis(
                solver="lsqr", covariance_estimator=LW, priors=[0.75, 0.25]
            )
        }
    )

    # replace this with graphical_lasso
    # gl = GraphicalLasso(alpha=0.2, mode='lars', max_iter = 1000)
    # modelList.update({"LDA GL": LinearDiscriminantAnalysis(solver='lsqr',
    #                                               covariance_estimator=gl,
    #                                               priors=[0.5,0.5])})
    # add priors as [0.5, 0.5]
    oa = OAS(store_precision=False, assume_centered=False)
    modelList.update(
        {
            "LDA OA": LinearDiscriminantAnalysis(
                solver="lsqr", covariance_estimator=oa, priors=[0.75, 0.25]
            )
        }
    )
    # modelList.update({
    #     "LgcRgs GL CV": GridSearchCV(
    #         LogisticGroupLasso(groups=groups,
    #                            scale_reg="inverse_group_size",
    #                            supress_warning=True,
    #                            subsampling_scheme=0.1,
    #                            n_iter=1500
    #                            ),
    #         param_grid = {'l1_reg': np.arange(0,1,0.05),
    #                       'group_reg': np.arange(0,1,0.05)
    #         )
    #     })

    modelList.update(
        {
            "NSC MH": GridSearchCV(
                _NSC.NearestCentroid(metric="manhattan"),
                param_grid={"shrink_threshold": np.arange(0.05, 1, 0.05)},
                cv=3,
            )
        }
    )
    # modelList.update({"NSC EC": GridSearchCV(NearestCentroid(metric='euclidean'),
    #                                          param_grid = {
    #                                              'shrink_threshold': np.arange(0.05,1,0.05)
    #                                              },
    #                                          cv=3)})

    modelList.update(
        {
            "NSC Custom EC": GridSearchCV(
                _NSC.NearestCentroid(metric="euclidean"),
                param_grid={"shrink_threshold": np.arange(0.05, 2, 0.05)},
                cv=3,
            )
        }
    )

    # modelList.update({"SVM Lin": GridSearchCV(SVC(kernel='linear'),
    #                                          param_grid = {
    #                                              'C': np.logspace(-10,3,base=2)
    #                                              },
    #                                          scoring='roc_auc')})
    #
    # modelList.update({"SVM Poly2": GridSearchCV(SVC(kernel='poly', degree=2),
    #                                          param_grid = {
    #                                              'C': np.logspace(-10,3,base=2)
    #                                              },
    #                                          scoring='roc_auc')})
    #
    # modelList.update({"SVM Poly3": GridSearchCV(SVC(kernel='poly', degree=3),
    #                                          param_grid = {
    #                                              'C': np.logspace(-10,3,base=2)
    #                                              },
    #                                          scoring='roc_auc')})
    #
    # modelList.update({"SVM Poly4": GridSearchCV(SVC(kernel='poly', degree=4),
    #                                          param_grid = {
    #                                              'C': np.logspace(-10,3,base=2)
    #                                              })})
    #
    # modelList.update({"SVM RBF": GridSearchCV(SVC(kernel='rbf'),
    #                                          param_grid = {
    #                                              'C': np.logspace(-10,3,base=2)
    #                                              })})

    modelList.update({"GNB": GaussianNB()})

    modelList.update({"GNB Prior": GaussianNB(priors=[0.8, 0.20])})

    modelList.update({"Tree": DecisionTreeClassifier()})

    # modelList.update({"CNB": ComplementNB()})
    # modelList.update({"KNN EC": GridSearchCV(KNeighborsClassifier(metric='euclidean'),
    #                                          param_grid = {
    #                                              'n_neighbors': [2, 3, 4, 5, 6, 7, 8],
    #                                              'weights': ['uniform','distance'],
    #                                              },
    #                                          scoring='roc_auc',
    #                                          cv=3)})

    # modelList.update({"KNN MH": GridSearchCV(KNeighborsClassifier(metric='manhattan'),
    #                                          param_grid = {
    #                                              'n_neighbors': [2, 3, 4, 5, 6, 7, 8],
    #                                              'weights': ['uniform','distance'],
    #                                              })})
    # modelList.update({"RNN EC": GridSearchCV(RadiusNeighborsClassifier(metric='euclidean'),
    #                                          param_grid = {
    #                                              'radius': np.arange(0.05,1,0.05),
    #                                              'weights': ['uniform','distance'],
    #                                              })})
    #
    # modelList.update({"RNN MH": GridSearchCV(RadiusNeighborsClassifier(metric='manhattan'),
    #                                          param_grid = {
    #                                              'radius': np.arange(0.05,1,0.05),
    #                                              'weights': ['uniform','distance'],
    #                                              })})

    return modelList


def getEnsembleModelFn() -> modelDict:
    modelList = {}

    modelList.update(
        {"RF": RandomForestClassifier(class_weight="balanced", max_depth=5)}
    )
    modelList.update({"Bagging": BaggingClassifier()})
    modelList.update({"Extra RF": ExtraTreesClassifier(class_weight="balanced")})
    modelList.update({"Ada": AdaBoostClassifier()})
    # what parameters should I vary? for xgboost
    # n_estimators, subsample, min_samples_leafs,
    modelList.update({"GB": GradientBoostingClassifier()})

    return modelList
