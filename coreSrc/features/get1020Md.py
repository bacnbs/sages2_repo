"""
Created on Aug 28, 2023

@author: mning

Not standard 10-20 EEG Montage. Duke just called it this.
"""
import logging

logger = logging.getLogger("root")


def get1020() -> list:
    chns_10_20 = [
        "Fp1",
        "Fp2",
        "F7",
        "F3",
        "Fz",
        "F4",
        "F8",
        "FC5",
        "FC1",
        "FC2",
        "FC6",
        "T7",
        "C3",
        "Cz",
        "C4",
        "T8",
        "CP5",
        "CP1",
        "CP2",
        "CP6",
        "P7",
        "P3",
        "Pz",
        "P4",
        "P8",
        "POz",
        "O1",
        "O2",
    ]

    return chns_10_20
