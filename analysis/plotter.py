#!/usr/bin/python3
'''
A python script designed to plot any two attributes from our datasets.
Useful for visualizing correlation

Currently only useful for the Portugal dataset

outputs graph as plot.png


=== Notes on non-numeric data ===
There exists non-numeric data that we wish to analyze.
Consider adding conversions to analyze such data

'''

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

# read csv and return a tuple.
# first element in tuple is labels, second element is a list of lists of the data
# returns a new DataSet object
def readCSV(filePath):
    labels = []
    with open(filePath, 'r') as f:
        reader = csv.reader(f)
        table = list(reader)
    labels = table[0]
    table = table[1:]
    # convert integer data
    for r, line in enumerate(table):
        for c, elem in enumerate(line):
            if elem.isdigit():
                table[r][c] = int(elem)
    return DataSet(table, labels)

# given a numpy list of values, return true if they are all numeric
def isDataNumberic(data):
    for item in data.tolist():
        if not type(item) is int:
            return False
    return True


def main(args):
    if len(args) != 3:
        print("use: plotter.py var-index-1 var-index-2")
        return
    fileName = portugalPath + 'student-mat.csv'
    dataSet = readCSV(fileName)

    # show the user which attributes they selected
    index1 = int(args[1])
    index2 = int(args[2])
    dataX = np.array([row[index1] for row in dataSet.table]) # independent variable
    dataY = np.array([row[index2] for row in dataSet.table]) # dependent variable
    labelX = dataSet.labels[index1]
    labelY = dataSet.labels[index2]
    print("independent index:",index1,"; label:",labelX)
    print("dependent index:",index2,"; label:",labelY)
    print(len(dataX))
    print(len(dataY))

    # check if data is numeric
    if not isDataNumberic(dataX):
        print("Error: independent not numeric")
        return
    if not isDataNumberic(dataY):
        print("Error: dependent not numeric")
        return

    # plot the data
    # plt.plot(dataX,dataY, 'ro')
    plt.plot(np.unique(dataX), np.poly1d(np.polyfit(dataX, dataY, 1))(np.unique(dataX)))
    fig, ax = plt.subplots()
    fit = np.polyfit(dataX,dataY, deg=1)
    ax.plot(dataX,fit[0] * dataX + fit[1], color='red')
    ax.scatter(dataX,dataY)
    plt.xlabel(labelX)
    plt.ylabel(labelY)
    plt.savefig(outFile)

    print("Success!")


if __name__ == '__main__':
    main(sys.argv)
