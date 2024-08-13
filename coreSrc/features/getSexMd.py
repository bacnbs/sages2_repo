"""
Created on Sep 27, 2023

@author: mning
"""
import logging

logger = logging.getLogger("root")


def getSex(op_eo) -> dict:
    # To add dict as column to pd.Dataframe:
    # df.loc[list(dic), 'a'] = pd.Series(dic)
    if op_eo:
        # feature store has 85 rows
        list_sex = {
            "001": "M",
            "002": "F",
            "003": "F",
            "004": "F",
            "005": "F",
            "006": "F",
            "007": "F",
            "008": "F",
            "009": "F",
            "010": "M",
            "011": "M",
            "012": "F",
            "013": "F",
            "014": "M",
            "015": "F",
            "016": "M",
            "018": "M",
            "019": "F",
            "020": "M",
            "021": "M",
            "022": "M",
            "023": "F",
            "025": "M",
            "027": "M",
            "028": "F",
            "029": "F",
            "030": "F",
            "031": "F",
            "032": "F",
            "033": "F",
            "034": "F",
            "035": "F",
            "036": "M",
            "037": "F",
            "038": "F",
            "039": "M",
            "040": "M",
            "041": "F",
            "043": "M",
            "044": "F",
            "045": "M",
            "047": "F",
            "048": "F",
            "049": "M",
            "050": "M",
            "051": "F",
            "052": "M",
            "053": "F",
            "054": "F",
            "055": "F",
            "056": "F",
            "057": "M",
            "059": "F",
            "060": "F",
            "061": "F",
            "062": "M",
            "063": "F",
            "064": "M",
            "065": "F",
            "066": "M",
            "067": "M",
            "068": "F",
            "069": "F",
            "070": "F",
            "071": "F",
            "072": "F",
            "073": "M",
            "074": "F",
            "075": "M",
            "076": "F",
            "077": "M",
            "078": "F",
            "079": "F",
            "080": "F",
            "082": "F",
            "083": "M",
            "084": "F",
            "085": "F",
            "086": "F",
            "087": "F",
            "088": "M",
            "089": "M",
            "090": "F",
            "091": "F",
            "092": "F",
        }
    elif not op_eo:
        list_sex = {
            "001": "M",
            "002": "F",
            "003": "F",
            "004": "F",
            "005": "F",
            "006": "F",
            "007": "F",
            "008": "F",
            "009": "F",
            "010": "M",
            "011": "M",
            "012": "F",
            "013": "F",
            "014": "M",
            "015": "F",
            "016": "M",
            "018": "M",
            "019": "F",
            "020": "M",
            "021": "M",
            "022": "M",
            "023": "F",
            "025": "M",
            "027": "M",
            "028": "F",
            "029": "F",
            "030": "F",
            "031": "F",
            "032": "F",
            "033": "F",
            "034": "F",
            "035": "F",
            "036": "M",
            "037": "F",
            "038": "F",
            "039": "M",
            "040": "M",
            "041": "F",
            "043": "M",
            "044": "F",
            "045": "M",
            "047": "F",
            "048": "F",
            "049": "M",
            "050": "M",
            "051": "F",
            "052": "M",
            "053": "F",
            "054": "F",
            "055": "F",
            "056": "F",
            "057": "M",
            "059": "F",
            "060": "F",
            "061": "F",
            "062": "M",
            "063": "F",
            "064": "M",
            "065": "F",
            "066": "M",
            "067": "M",
            "068": "F",
            "069": "F",
            "070": "F",
            "071": "F",
            "072": "F",
            "073": "M",
            "074": "F",
            "075": "M",
            "076": "F",
            "077": "M",
            "078": "F",
            "079": "F",
            "080": "F",
            "082": "F",
            "083": "M",
            "084": "F",
            "085": "F",
            "086": "F",
            "087": "F",
            "088": "M",
            "089": "M",
            "090": "F",
            "091": "F",
            "092": "F",
            "017": "M",
            "026": "F",
            "042": "M",
        }
    return list_sex


def getSexDuke() -> dict:
    # To add dict as column to pd.Dataframe:
    # df.loc[list(dic), 'a'] = pd.Series(dic)
    dict_sex = {
        "0115": "F",
        "0182": "F",
        "0190": "F",
        "0192": "M",
        "023": "M",
        "024": "M",
        "025": "M",
        "027": "M",
        "028": "M",
        "032": "F",
        "033": "F",
        "034": "M",
        "038": "F",
        "040": "M",
        "043": "M",
        "044": "F",
        "045": "M",
        "046": "F",
        "047": "M",
        "049": "F",
        "050": "F",
        "051": "F",
        "056": "F",
        "057": "F",
        "059": "F",
        "061": "F",
        "062": "F",
        "064": "F",
        "066": "M",
        "067": "F",
        "068": "M",
        "069": "F",
        "074": "M",
        "077": "F",
        "079": "F",
        "083": "M",
        "085": "M",
        "092": "M",
        "093": "M",
        "094": "F",
        "097": "F",
        "098": "F",
        "102": "M",
        "110": "M",
        "112": "F",
        "113": "F",
        "185": "M",
        "193": "M",
        "194": "M",
        "195": "F",
        "198": "M",
        "199": "M",
        "201": "M",
    }
    return dict_sex
