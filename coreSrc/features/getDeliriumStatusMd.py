"""
Created on Nov 9, 2023

@author: mning
"""
from typing import Optional


def getDeliriumStatus(op_SAGES) -> Optional[list[str]]:
    if op_SAGES == 1 or op_SAGES == "SAGES":
        return [
            "003",
            "008",
            "011",
            "013",
            "023",
            "028",
            "050",
            "055",
            "056",
            "060",
            "073",
            "082",
        ]
    elif op_SAGES == 0 or op_SAGES == "DUKE":
        return ["0192", "025", "040", "050", "066", "193"]
    else:
        return None
