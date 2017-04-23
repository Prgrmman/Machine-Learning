#!/usr/bin/python3

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import csv
import sys
import numpy as np
from regressions import handleNominal
'''
this is what I'm reading to get started
http://www.statisticssolutions.com/mlr/

'''

# Paths to the different datasets
middle_east_path = '../data/middle east/middle_east.csv' 
portugal_math_path = '../data/portugal/student-mat.csv'
portugal_por_path = '../data/portugal/student-por.csv'


# Reads a a csv and returns numpy 2d array
def readData(path):
    with open(path) as f:
        reader = csv.reader(f)
        table = list(reader)
        return np.array(table)


# sanatize input from data file
# if nominal, feaure is converted to matrix of dummy variables
def parseData(feature):
    """
    feature: (numpy.array) list of feature data
    """
    try:
        feature = feature.astype(np.int)
    except ValueError:
        print("Whoops")
    return feature.T

'''
Runs the multinomial logistic regression

'''
#TODO I think the feature dimensions are wrong somehow
def runLogistic(matrix, features_list_indexes, target_index, penalty_term):
    """
    matrix: numpy 2d data marix
    features_list_indexes: (list of ints) indexes of selected features
    target_index: (int) index of target class
    penalty_term: (float) strength of L2 penalty
    """
    features = []
    for index in features_list_indexes:
        feature = matrix[1:, index]
        features.append(feature)
    
    # build dataset again
    data = features[0].reshape(len(matrix)-1, 1)
    for feature in features[1:]:
        feature = feature.reshape(len(matrix)-1, 1)
        data = np.hstack((data, feature))
    target = matrix[1:, target_index].reshape(len(matrix)-1, 1)

    data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, target, test_size = 0.2)

    
    


def main(args):
    middle_east_matrix = readData(middle_east_path)
    portugal_math_matrix = readData(portugal_math_path)
    portugal_por_matrix = readData(portugal_por_path)

    runLogistic(middle_east_matrix, [0,1,2], 16, 0)
    


if __name__ == '__main__':
    main(sys.argv)
