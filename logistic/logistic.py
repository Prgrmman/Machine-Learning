#!/usr/bin/python3


def warn(*args,**kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.model_selection import cross_val_score
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
verbose = False # turn off extra print statements


# calls print if verbose is True
def verbose_print(*args, **kwargs):
    if verbose:
        print(*args, **kwargs)
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

# turns a numeric variable into nominal feature
def makeNominal(feature, num_levels):
    """
    feature: (numpy array)
    num_levels: the number of levels you wish to split the value into
    """
    feature = feature.astype(np.int)
    max_value = max(list(feature))

    levels = [0]
    for i in range(num_levels - 1):
        value = max_value * ((i+1)/(num_levels))
        levels.append(value)

    for i, value in enumerate(feature):
        for j in range(num_levels-1):
            if levels[j] <= value < levels[j+1]:
                feature[i] = j
        j = num_levels - 1
        if value >= levels[j]:
            feature[i] = j
    return feature


'''
Runs the multinomial logistic regression
evaluates using 10-fold cross validation

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
    accuracy = cross_val_score(logreg, data, target, cv=10) 
    return accuracy

# runs 20 tests and returns the average
def averageTest(matrix, features_list_indexes, target_index, penalty):
    accs = runLogistic(matrix, features_list_indexes, target_index, penalty)
    average = sum(accs) / len(accs)
    print ("Accuracy with penalty", penalty, "->",average)

# prints usage message and exits with error
def usuage(name):
    print("Usage:" + name, "middle_east/por_math/por_lang reg_constant 5/10/all")
    sys.exit(1)


def main(args):

    # process arguments 
    if len(args) != 4:
        usuage(args[0])
    chosen_data = args[1]
    reg_constant = float(args[2])
    which_features = args[3]

    if not chosen_data in ["middle_east", "por_math", "por_lang"]:
        print(chosen_data+ ":", "not valid data set name")
        usuage(args[0])
    if not which_features in ["5", "10", "all"]:
        print(which_features+ ":", "not valid feature option")
        usuage(args[0])

    middle_east_matrix = readData(middle_east_path)
    portugal_math_matrix = readData(portugal_math_path)
    portugal_por_matrix = readData(portugal_por_path)

    if chosen_data == "middle_east":
        verbose_print("running middle east")
        verbose_print("\nSelecting features:")
        indexes = list(range(16))
        for index in indexes:
            label = middle_east_matrix[0, index]
            verbose_print(label)
        
        '''
        running on middle east data
        run average accuracy with 0.5 L2 penalty
        '''
        averageTest(middle_east_matrix, indexes, 16, reg_constant)

    elif chosen_data == "por_math":
        scores = portugal_math_matrix[1:,32]
        scores = makeNominal(scores,3)
        portugal_math_matrix[1:,32] = scores
        if which_features == "all":
            '''
            Running portugal math data with no feature selection
            Don't select index 0: it just contains which school they went too
            '''
            verbose_print("running portugal math scores at 3 levels")
            verbose_print("\nSelecting features:")
            indexes = list(range(1,30))
            for index in indexes:
                label = portugal_math_matrix[0,index]
                verbose_print(label)
            averageTest(portugal_math_matrix, indexes, 32, reg_constant)

        elif which_features == "10":
            '''
            Running portugal math data with features selected from 10% significance level
            '''
            verbose_print("running portugal math scores at 3 levels with f-tested features at 10% sig level")
            verbose_print("\nSelecting features:")
            indexes = [1,2,3,6,7,8,12,13,14,17,20,21,22,25]
            for index in indexes:
                label = portugal_math_matrix[0,index]
                verbose_print(label)

            averageTest(portugal_math_matrix, indexes, 32, reg_constant)
        elif which_features == "5":
            '''
            Running portugal math scores at 3 levels with f-tested features at 5% significance level
            '''
            verbose_print("running portugal math scores at 3 levels with f-tested features at 5% sig level")
            verbose_print("\nSelecting features:")
            indexes = [1,2,3,6,7,8,12,14,17,20,22,25]
            for index in indexes:
                label = portugal_math_matrix[0,index]
                verbose_print(label)
            averageTest(portugal_math_matrix, indexes, 32, reg_constant)

    elif chosen_data == "por_lang":
        scores = portugal_por_matrix[1:,32]
        scores = makeNominal(scores,3)
        portugal_por_matrix[1:,32] = scores
        if which_features == "all":
            verbose_print("running portugal language scores at 3 levels")
            verbose_print("\nSelecting features:")
            indexes = list(range(1,30))
            for index in indexes:
                label = portugal_por_matrix[0,index]
                verbose_print(label)
            averageTest(portugal_por_matrix, indexes, 32, reg_constant)
        elif which_features == "10":
            verbose_print("running portugal language scores at 3 levels with f-tested features at 10% sig level")
            verbose_print("\nSelecting features:")
            indexes = [0,1,2,3,6,7,8,9,12,11,12,13,14,15,20,21,22,24,25,26,27,28,29]
            for index in indexes:
                label = portugal_por_matrix[0,index]
                verbose_print(label)
            averageTest(portugal_por_matrix, indexes, 32, reg_constant)

        elif which_features == "5":
            verbose_print("running portugal language scores at 3 levels with f-tested features at 5% sig level")
            verbose_print("\nSelecting features:")
            indexes = [0,1,2,3,6,7,8,9,10,12,13,14,20,22,24,25,26,27,28,29]
            for index in indexes:
                label = portugal_por_matrix[0,index]
                verbose_print(label)
            averageTest(portugal_por_matrix, indexes, 32, reg_constant)

if __name__ == '__main__':
    main(sys.argv)
