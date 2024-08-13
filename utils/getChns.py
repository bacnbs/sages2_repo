"""
Created on May 31, 2023

@author: mning
"""

chnId = list[int]
chnNames = list[str]
chnInfo = tuple[chnId, chnNames]
bipolarChnLists = tuple[chnNames, ...]


def getFrontChns(chn_names: list) -> chnInfo:
    id_Fp1 = chn_names.index("Fp1")
    id_Fp2 = chn_names.index("Fp2")
    id_AF3 = chn_names.index("AF3")
    id_AF4 = chn_names.index("AF4")
    id_AF7 = chn_names.index("AF7")
    id_AF8 = chn_names.index("AF8")
    id_F7 = chn_names.index("F7")
    id_F8 = chn_names.index("F8")
    id_F5 = chn_names.index("F5")
    id_F6 = chn_names.index("F6")
    id_F3 = chn_names.index("F3")
    id_F4 = chn_names.index("F4")
    id_F1 = chn_names.index("F1")
    id_F2 = chn_names.index("F2")

    id_List = [
        id_Fp1,
        id_Fp2,
        id_AF3,
        id_AF4,
        id_AF7,
        id_AF7,
        id_AF8,
        id_F7,
        id_F8,
        id_F5,
        id_F6,
        id_F3,
        id_F4,
        id_F1,
        id_F2,
    ]
    name_List = [
        "Fp1",
        "Fp2",
        "AF3",
        "AF4",
        "AF7",
        "AF8",
        "F7",
        "F8",
        "F5",
        "F6",
        " F3",
        "F4",
        "F1",
        "F2",
    ]

    return id_List, name_List


def getOccipitalChns(chn_names: list) -> chnInfo:
    id_Oz = chn_names.index("Oz")
    id_O1 = chn_names.index("O1")
    id_O2 = chn_names.index("O2")
    id_PO7 = chn_names.index("PO7")
    id_PO8 = chn_names.index("PO8")
    id_PO3 = chn_names.index("PO3")
    id_PO4 = chn_names.index("PO4")
    id_POz = chn_names.index("POz")
    id_P7 = chn_names.index("P7")
    id_P8 = chn_names.index("P8")
    id_P5 = chn_names.index("P5")
    id_P6 = chn_names.index("P6")
    id_P3 = chn_names.index("P3")
    id_P4 = chn_names.index("P4")
    id_P1 = chn_names.index("P1")
    id_P2 = chn_names.index("P2")

    id_List = [
        id_Oz,
        id_O1,
        id_O2,
        id_PO7,
        id_PO8,
        id_PO3,
        id_PO4,
        id_POz,
        id_P7,
        id_P8,
        id_P5,
        id_P6,
        id_P3,
        id_P4,
        id_P1,
        id_P2,
    ]
    name_List = [
        "Oz",
        "O1",
        "O2",
        "PO7",
        "PO8",
        "PO3",
        "PO4",
        "POz",
        "P7",
        "P8",
        "P5",
        "P6",
        "P3",
        "P4",
        "P1",
        "P2",
    ]

    return id_List, name_List


def getCentralChns(chn_names: list) -> chnInfo:
    id_Cz = chn_names.index("Cz")
    id_FCz = chn_names.index("FCz")
    id_CPz = chn_names.index("CPz")
    id_FC1 = chn_names.index("FC1")
    id_FC2 = chn_names.index("FC2")
    id_FC3 = chn_names.index("FC3")
    id_FC4 = chn_names.index("FC4")
    id_C1 = chn_names.index("C1")
    id_C2 = chn_names.index("C2")
    id_C3 = chn_names.index("C3")
    id_C4 = chn_names.index("C4")
    id_CP1 = chn_names.index("CP1")
    id_CP2 = chn_names.index("CP2")
    id_CP3 = chn_names.index("CP3")
    id_CP4 = chn_names.index("CP4")

    id_List = [
        id_Cz,
        id_FCz,
        id_CPz,
        id_FC1,
        id_FC2,
        id_FC3,
        id_FC4,
        id_C1,
        id_C2,
        id_C3,
        id_C4,
        id_CP1,
        id_CP2,
        id_CP3,
        id_CP4,
    ]
    name_List = [
        "Cz",
        "FCz",
        "CPz",
        "FC1",
        "FC2",
        "FC3",
        "FC4",
        "C1",
        "C2",
        "C3",
        "C4",
        "CP1",
        "CP2",
        "CP3",
        "CP4",
    ]

    return id_List, name_List


def getAnodeCathode() -> bipolarChnLists:
    """
    May want to check this with other ppl

    Idea: for bipolar reference, artifacts should be common to both electrodes
    but so are neurophysiological signals of interested. Whether the difference
    are actually of interested is an area of research.

    For equal gain combining diversity, external noises in each electrodes are
    independent of each other. However, will amplify both physiological artifacts
    and neurophysiological signals of interested.

    So try both!

    38 channels for now.

    Can add more if needed.

    Some helper functions from mne library for future usage:
    use mne.io.pick.pick_channels
    1.
    sel = pick_channels(info['ch_names'], myinclude, exclude)

    2.
    ch_type = 'auto'
    ch_type = _get_ch_type(inst, ch_type)
    ch_dict = {**{type_: True for type_ in ch_type},
               'meg': False, 'ref_meg': False}
    eeg_idx = pick_types(inst.info, **ch_dict)

    """

    anode = [
        "Fp1",
        "AF3",
        "F3",
        "FC3",
        "C3",
        "CP3",
        "P3",
        "PO3",
        "Fp1",
        "AF7",
        "F7",
        "FT7",
        "T7",
        "TP7",
        "P7",
        "PO7",
        "Fz",
        "FCz",
        "Cz",
        "CPz",
        "Pz",
        "POz",
        "Fp2",
        "AF4",
        "F4",
        "FC4",
        "C4",
        "CP4",
        "P4",
        "PO4",
        "Fp2",
        "AF8",
        "F8",
        "FT8",
        "T8",
        "TP8",
        "P8",
        "PO8",
        "F1",
        "FC1",
        "C1",
        "CP1",
        "F2",
        "FC2",
        "C2",
        "CP2",
        "F5",
        "FC5",
        "C5",
        "CP5",
        "F6",
        "FC6",
        "C6",
        "CP6",
    ]
    cathode = [
        "AF3",
        "F3",
        "FC3",
        "C3",
        "CP3",
        "P3",
        "PO3",
        "O1",
        "AF7",
        "F7",
        "FT7",
        "T7",
        "TP7",
        "P7",
        "PO7",
        "O1",
        "FCz",
        "Cz",
        "CPz",
        "Pz",
        "POz",
        "Oz",
        "AF4",
        "F4",
        "FC4",
        "C4",
        "CP4",
        "P4",
        "PO4",
        "O2",
        "AF8",
        "F8",
        "FT8",
        "T8",
        "TP8",
        "P8",
        "PO8",
        "O2",
        "FC1",
        "C1",
        "CP1",
        "P1",
        "FC2",
        "C2",
        "CP2",
        "P2",
        "FC5",
        "C5",
        "CP5",
        "P5",
        "FC6",
        "C6",
        "CP6",
        "P6",
    ]
    chns_drop = ["P1", "P5", "P6", "P2"]

    return anode, cathode, chns_drop
