"""
Created on Aug 25, 2023

@author: mning

For GCP, round to the nearest 10th
"""
import logging

logger = logging.getLogger("root")


def getGCP(op_eo) -> dict:
    # To add dict as column to pd.Dataframe:
    # df.loc[list(dic), 'a'] = pd.Series(dic)
    if op_eo:
        # feature store has 85 rows
        moca_scrs = {
            "001": 49.8,
            "002": 41.5,
            "003": 29.5,
            "004": 55.7,
            "005": 65.7,
            "006": 65.8,
            "007": 63.9,
            "008": 38.0,
            "009": 46.9,
            "010": 67.4,
            "011": 64.2,
            "012": 65.8,
            "013": 29.7,
            "014": 48.2,
            "015": 60.0,
            "016": 60.8,
            "018": 62.6,
            "019": 58.1,
            "020": 66.0,
            "021": 57.3,
            "022": 54.8,
            "023": 69.6,
            "025": 70.4,
            "027": 54.3,
            "028": 53.9,
            "029": 59.2,
            "030": 68.8,
            "031": 60.7,
            "032": 63.9,
            "033": 54.7,
            "034": 40.7,
            "035": 64.1,
            "036": 57.8,
            "037": 67.1,
            "038": 63.8,
            "039": 50.2,
            "040": 54.6,
            "041": 65.0,
            "043": 69.9,
            "044": 64.7,
            "045": 57.7,
            "047": 68.8,
            "048": 75.2,
            "049": 66.1,
            "050": 55.7,
            "051": 68.6,
            "052": 55.6,
            "053": 72.3,
            "054": 62.2,
            "055": 70.2,
            "056": 41.5,
            "057": 54.3,
            "059": 67.3,
            "060": 58.6,
            "061": 75.5,
            "062": 51.1,
            "063": 77.3,
            "064": 59.6,
            "065": 61.0,
            "066": 59.0,
            "067": 69.3,
            "068": 62.7,
            "069": 69.0,
            "070": 71.4,
            "071": 66.4,
            "072": 53.3,
            "073": 48.9,
            "074": 67.0,
            "075": 68.0,
            "076": 68.0,
            "077": 52.3,
            "078": 69.8,
            "079": 59.2,
            "080": 66.7,
            "082": 47.9,
            "083": 75.8,
            "084": 53.7,
            "085": 57.4,
            "086": 66.3,
            "087": 64.1,
            "088": 48.8,
            "089": 49.0,
            "090": 69.9,
            "091": 55.3,
            "092": 51.6,
        }
    elif not op_eo:
        moca_scrs = {
            "001": 49.8,
            "002": 41.5,
            "003": 29.5,
            "004": 55.7,
            "005": 65.7,
            "006": 65.8,
            "007": 63.9,
            "008": 38.0,
            "009": 46.9,
            "010": 67.4,
            "011": 64.2,
            "012": 65.8,
            "013": 29.7,
            "014": 48.2,
            "015": 60.0,
            "016": 60.8,
            "018": 62.6,
            "019": 58.1,
            "020": 66.0,
            "021": 57.3,
            "022": 54.8,
            "023": 69.6,
            "025": 70.4,
            "027": 54.3,
            "028": 53.9,
            "029": 59.2,
            "030": 68.8,
            "031": 60.7,
            "032": 63.9,
            "033": 54.7,
            "034": 40.7,
            "035": 64.1,
            "036": 57.8,
            "037": 67.1,
            "038": 63.8,
            "039": 50.2,
            "040": 54.6,
            "041": 65.0,
            "043": 69.9,
            "044": 64.7,
            "045": 57.7,
            "047": 68.8,
            "048": 75.2,
            "049": 66.1,
            "050": 55.7,
            "051": 68.6,
            "052": 55.6,
            "053": 72.3,
            "054": 62.2,
            "055": 70.2,
            "056": 41.5,
            "057": 54.3,
            "059": 67.3,
            "060": 58.6,
            "061": 75.5,
            "062": 51.1,
            "063": 77.3,
            "064": 59.6,
            "065": 61.0,
            "066": 59.0,
            "067": 69.3,
            "068": 62.7,
            "069": 69.0,
            "070": 71.4,
            "071": 66.4,
            "072": 53.3,
            "073": 48.9,
            "074": 67.0,
            "075": 68.0,
            "076": 68.0,
            "077": 52.3,
            "078": 69.8,
            "079": 59.2,
            "080": 66.7,
            "082": 47.9,
            "083": 75.8,
            "084": 53.7,
            "085": 57.4,
            "086": 66.3,
            "087": 64.1,
            "088": 48.8,
            "089": 49.0,
            "090": 69.9,
            "091": 55.3,
            "092": 51.6,
            "017": 53.9,
            "026": 58.9,
            "042": 61.9,
        }
    return moca_scrs
