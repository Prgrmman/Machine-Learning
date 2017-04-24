#!/usr/bin/python3

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import accuracy_score
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
        return handleNominal(feature)
    return feature.reshape(len(feature), 1)

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
    data = parseData(features[0])
    for feature in features[1:]:
        feature = parseData(feature)
        data = np.hstack((data, feature))
    target = matrix[1:, target_index]

    data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, target, test_size = 0.33)

    logreg = LogisticRegression(C=penalty_term, multi_class='multinomial', solver='newton-cg') 
    logreg.fit(data_train, target_train)

    predictions = logreg.predict(data_test)
    accuracy = accuracy_score(target_test, predictions)
    return accuracy

def main(args):
    middle_east_matrix = readData(middle_east_path)
    portugal_math_matrix = readData(portugal_math_path)
    portugal_por_matrix = readData(portugal_por_path)

    print("running middle east")
    print("Selecting features:\n")
    indexes = list(range(16))
    for index in indexes:
        label = middle_east_matrix[0, index]
        print(label)
    
    # run average accuracy with 0.5 L2 penalty
    total = 0
    for i in range(20):
        accuracy = runLogistic(middle_east_matrix, indexes, 16, 0.5)
        print("Accuracy:", accuracy)
        total += accuracy
    print("Average accuracy for 20 iterations:" ,total / 20)
    


if __name__ == '__main__':
    main(sys.argv)
