#!/usr/bin/python3

import sys
import csv
import scipy as stats
import statsmodels.api as sm
import numpy as np
from regressions import handleNominal


'''
Globals
'''
dataPath = '../data/middle east/middle_east.csv'

def isNumeric(X):
    """
    Parameters:
        X: list of data in string form
    Description:
       returns false if X contains any strings 
    """
    for item in list(X):
        if not item.isdigit():
            return False
    return True


# There are 4 cases:
#     X nominal, Y nominal -> chi-square test
#     X nominal, Y continuous or vice versa -> f-test
#     X continuous, Y continuous -> f-test
def dataTest(X, Y):
    """
    Parameters:
        data1: list of values (string form)
        data2: list of values (string form)
    Description:
        attempts to determine if variables are related
    Returns:
        p-value of chosen test
    """
    if isNumeric(X) and isNumeric(Y):
        return fTest(X,Y)
    elif isNumeric(X) and not isNumeric(Y):
        return fTest(Y,X)
    elif isNumeric(Y) and not isNumeric(X):
        return fTest(X,Y)
    elif not isNumeric(X) and not isNumeric(Y):
        return chiSquareTest(X,Y)


# constructs linear regression and performs f-test
# X and Y are lists of values (string form)
# X can be nominal, but Y will always be numeric
def fTest(X,Y):

    pass

# does chi-square test for independence
# X and Y are lists of values (string form)
def chiSquareTest(X,Y):
    pass


# returns a numpy matrix with dummy values assigned
def handleDummy(X):
    pass


def main(args):
    print(dataPath)
    with open(dataPath, 'r') as f:
        reader = csv.reader(f)
        table = list(reader)
        matrix = np.array(table)
    print(matrix[0].size)

    X = matrix[1:, 11]
    Y = matrix[1:, 16]
    # print(isNumeric(X))
    # print(X)

if __name__ == '__main__':
    main(sys.argv)
