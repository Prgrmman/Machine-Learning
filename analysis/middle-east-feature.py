#!/usr/bin/python3

import sys
import csv
import scipy as stats
import statsmodels.api as sm
import numpy as np
from regressions import handleNominal
from sklearn.feature_selection import chi2


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


# maps nominal to integer
# different than handleNominal
def nominalMap(feature):
    
    valueList = list(set(list(feature)))
    valueList.sort()
    return [valueList.index(value) + 1 for value in list(feature)]


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
    print("using f-test on linear model")
    if not isNumeric(X):
        X = handleNominal(X)
    X = sm.add_constant(X)
    Y = Y.astype(np.float)
    olsModel = sm.OLS(Y,X)
    est = olsModel.fit()
    return est.f_pvalue


# does chi-square test for independence
# X and Y are lists of values (string form)
def chiSquareTest(X,Y):
    X = nominalMap(X)
    Y = nominalMap(Y)
    print("First is", X[0])
    data = [X,Y]
    results = chi2(X,Y)
    return results[1]



def main(args):
    if len(args) != 2:
        print("Usage: ./middle-east-feature.py sig-level")
        return
    sigLevel = float(args[1])
    with open(dataPath, 'r') as f:
        reader = csv.reader(f)
        table = list(reader)
        matrix = np.array(table)

    selectedFeatures = []
    for i in range(15):
        X = matrix[1:, i]
        Y = matrix[1:, 16] # 16 is the index of the target in the dataset
        labelX = matrix[0,i]
        print("Testing", labelX)
        pvalue = dataTest(X,Y)
        print(pvalue)
        if pvalue > sigLevel:
            print("Failed")
        else:
            selectedFeatures.append(labelX)

    print("Selected", len(selectedFeatures), "feature(s)")
    print(selectedFeatures)

if __name__ == '__main__':
    main(sys.argv)
