"""
Created on May 11, 2023

@author: mning
"""


class logFilter(object):
    def __init__(self, level):
        self.__level = level

    def filter(self, logRecord):
        return logRecord.levelno >= self.__level
