"""
Created on Jan 22, 2024

@author: mning
"""
from typing import List

# color palette generated from one color here:
# https://mycolor.space/
# OLD
# clr_list_drk = ['#65ABEC',
#             '#FFE9D0',
#             '#A4CB9B',
#             '#D79186',
#             '#C1A4D8',
#             '#C4887C',
#             '#8E587C']


def get_def_clr() -> List:
    default_clr_list = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#F9F871",
    ]
    return default_clr_list


def get_clr_drk() -> List:
    clr_list_drk = [
        "#3C4856",
        "#402E32",
        "#3F4A3C",
        "#56423E",
        "#300A58",
        "#3A001E",
        "#594F6E",
        "#C3A689",
    ]
    return clr_list_drk


def get_clr_lght() -> List:
    clr_list_lght = [
        "#65ABEC",
        "#BBA79C",
        "#A4CB9B",
        "#D79186",
        "#C1A4D8",
        "#C4887C",
        "#D8BFD8",
        "#947046",
    ]
    return clr_list_lght
