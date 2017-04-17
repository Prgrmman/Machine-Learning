#!/usr/bin/python3
'''
A python script designed to plot any two attributes from Portugal dataset
Useful for visualizing correlation


outputs graph as plot.png


=== Notes on non-numeric data ===
There exists non-numeric data that we wish to analyze.
Note, some data does not not start at 0 because of shared keys
For data that has yes,no:
-> no = 0
-> yes = 1

sex:
-> M = 0
-> F = 1

address:
-> "U" urban = 0
-> "R" rural = 1

famsize:
-> "LE3" = 0
-> "GT3" = 1

Pstatus:
-> "T" together = 0
-> "A" apart = 1

Mjob:
-> teacher = 0
-> health = 1
-> services = 2
-> at_home = 3
-> other = 4

Fjob: 
-> teacher = 0
-> health = 1
-> services = 2
-> at_home = 3
-> other = 4

reason:
home = 1
reputation = 2
course = 3
other = 4

gaurdian:
-> "mother" = 2
-> "father" = 3
-> "other" = 4
'''

# TODO add normalization

import matplotlib.pyplot as plt
import csv
import sys
import numpy as np


# Class definitions

class DataSet:
    """
    Fields:
        self.table: the data
        self.labels: the data labels
        self.labelDict: label dictionary to table index
    """
    def __init__(self, table, labels):
        """
        table: list of lists
        labels: list of strings
        """
        self.table = table
        self.labels = labels
        self.labelDict = {}
        for i, label in enumerate(labels):
            self.labelDict[label] = i



# Global definitions
outFile = 'plot.png'
portugalPath = '../data/portugal/'
nominalDict = {'M': 0, 'F': 1, 'U': 0, 'R': 1,'LE3':0, 'GT3':1, 'T':0, 'A':1, 'teacher': 0, 'health':1,
        'services':2, 'at_home':3, 'other': 4, 'home':1, 'reputation':2, 'course': 3, 'mother':2, 'father':3,
        'no':0, 'yes': 1}

# read csv and return a tuple.
# first element in tuple is labels, second element is a list of lists of the data
# returns a new DataSet object
def readCSV(filePath):
    labels = []
    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        table = list(reader)
    # remove first column from table
    for i in range(len(table)):
        table[i] = table[i][1:]
    labels = table[0]
    table = table[1:]
    # convert integer data
    for r, line in enumerate(table):
        for c, elem in enumerate(line):
            if elem.isdigit():
                table[r][c] = int(elem)
            # handle non numeric data
            else: 
                table[r][c] = nominalDict.get(elem, -1)
    return DataSet(table, labels)

# given a numpy list of values, return true if they are all numeric
def isDataNumberic(data):
    for item in data.tolist():
        if not type(item) is int:
            return False
    return True

# correlate
def correlate(index1, index2, setName = 'student-mat.csv'):
    fileName = portugalPath + setName
    dataSet = readCSV(fileName)
    dataX = np.array([row[index1] for row in dataSet.table]) # independent variable
    dataY = np.array([row[index2] for row in dataSet.table]) # dependent variable
    labelX = dataSet.labels[index1]
    labelY = dataSet.labels[index2]
    print("independent index:",index1,"; label:",labelX)
    print("dependent index:",index2,"; label:",labelY)

    # check if data is numeric
    if not isDataNumberic(dataX):
        print("Error: independent not numeric")
        return
    if not isDataNumberic(dataY):
        print("Error: dependent not numeric")
        return

    # plot the data
    fig, ax = plt.subplots()
    fit = np.polyfit(dataX,dataY, deg=1)
    ax.plot(dataX,fit[0] * dataX + fit[1], color='red')
    ax.scatter(dataX,dataY)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.savefig(outFile)

    # print the Pearson correlation coefficient
    print(np.corrcoef(dataX,dataY)[0,1])

    print("Success!")


def main(args):
    if len(args) != 3:
        print("use: plotter.py var-index-1 var-index-2")
        return

    # show the user which attributes they selected
    index1 = int(args[1])
    index2 = int(args[2])
    correlate(index1,index2)


if __name__ == '__main__':
    main(sys.argv)
