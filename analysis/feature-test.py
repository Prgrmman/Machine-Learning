#!/usr/bin/python
import csv
import numpy as np
import statsmodels.api as sm
import sys
from scipy import stats


'''
Notes on dummy variables
Dumpy variables will only be created for nominal features of more than one class
Otherwise, default mapping is 0 and 1
'''

'''
globals
'''
# mapping for binary variables
binaryDict = {'GP':0,'MS':1 ,'M': 0, 'F': 1, 'U': 0, 'R': 1,'LE3':0, 'GT3':1, 'T':0, 'A':1,'no':0, 'yes': 1}

# list of tuples. First element is index, second is the number of values the variable takes
nominalPairs = [8,9,10,11]
portugalPath = '../data/portugal/'

'''
Functions
'''
# function that handles nominal variables
# second parameter should be passed g3_score
# third parameter is data matrix
# returns p-value from ftest
def handleNominal(index, dependent, matrix):
    feature = matrix[1:, index]
    numFeatrues = feature.size
    valueSet = set(list(feature))
    valueList = list(valueSet)
    valueList.sort()
    # create dummy variables. For k possible values, create k-1 values
    dummys = [[0] * numFeatrues for i in range(len(valueList)-1)]

    # go through all values
    for i,value in enumerate(list(feature)):
        valueIndex = valueList.index(value)
        if valueIndex >= len(dummys):
            continue
        dummys[valueIndex][i] = 1
        
    dummys = np.array(dummys)
    dummys = dummys.T
    dummys = dummys.astype(np.float)
    X = sm.add_constant(dummys)
    olsModel = sm.OLS(dependent, X)
    est = olsModel.fit()
    return est.f_pvalue
    

'''
Main
'''

def main(args):
    if len(args) != 3:
        print("usage: ./feature-test.py dataset sig-level (percent)")
        print("Example: run f-tests on student-mat.csv with 5.6 percent significance level")
        print("./feature-test.py student-mat.csv 5.6")
        return


    filepath = portugalPath + args[1]
    sigLevel = float(args[2]) # user set significance level
    sigLevel *= 10**-2        # normalize percent  
    print(sigLevel)
    selectedFeatures = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        table = list(reader)
        for i, line in enumerate(table):
            for j, elem in enumerate(line):
                if elem in binaryDict:
                    table[i][j] = binaryDict[elem]
        matrix = np.array(table)
    #get g3 score and indep
    g3_score = matrix[1:,32]
    g3_score = g3_score.astype(np.float)
    g3_score = g3_score.reshape(-1,1)


    # main loop
    for i in range(matrix[0].size - 3):
        label = matrix[0,i]
        indep = matrix[1:,i]
        print("Testing", label)
        if i in nominalPairs:
            print("Nominal!")
            pvalue = handleNominal(i,g3_score, matrix)
            print(pvalue)
            if pvalue > sigLevel:
                print("Failed!")
            else:
                selectedFeatures.append(label)
            continue
        indep = indep.astype(np.float)
        indep = indep.reshape(-1,1)

        # prints summary of of regression
        X = sm.add_constant(indep)
        olsModel = sm.OLS(g3_score, X)
        est = olsModel.fit()
        #print(est.summary())
        pvalue = est.f_pvalue
        print(pvalue)
        if pvalue > sigLevel:
            print("Failed!")
        else:
            selectedFeatures.append(label)

    # print summary 
    print("Selected", len(selectedFeatures), "feature(s)")
    print(selectedFeatures)

if __name__ == '__main__':
    main(sys.argv)
    
