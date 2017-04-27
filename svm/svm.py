#!/usr/bin/python3

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import csv
import random
import sys

#paths
middle_east = '../data/middle east/middle_east.csv'
portugal_math = '../data/portugal/student-mat.csv'
portugal_por = '../data/portugal/student-por.csv'

#dicts
midDict = {'F': 0, 'M': 1, 'S': 1, 'KW': 0, 'lebanon': 1, 'KW': 1,
'Egypt': 2, 'Saudi Arabia': 3, 'USA': 4, 'Jordan': 5, 'venezuala': 6,
'Iran': 7, 'Tunis': 8, 'Syria': 9, 'Morocco': 10, 'Palestine': 11,
'Iraq':12, 'Lybia': 13, 'KuwaIT': 0, 'SaudiArabia': 3, 'lowerlevel': 0,
'MiddleSchool': 1, 'HighSchool': 2, 'G-01': 1, 'G-02': 2, 'G-03': 3,
'G-04': 4, 'G-05': 5, 'G-06': 6, 'G-07': 7, 'G-08': 8, 'G-09': 9,
'G-10': 10, 'G-11': 11, 'G-12': 12, 'A': 0, 'B': 1, 'C': 2, 'IT': 0,
'Math':1, 'Arabic': 2, 'Science': 3, 'English': 4, 'Quaran': 5,
'Spanish': 6, 'French': 7, 'History': 8, 'Biology': 9, 'Chemistry': 10,
'Geology': 11, 'Mum': 0, 'Father': 1, 'No': 0, 'Yes': 1, 'Bad': 0,
'Good': 1, 'Under-7': 0, 'Above-7': 1, 'L': 0, 'H': 2}

porDict = {'M': 0, 'F': 1, 'U': 0, 'R': 1,'LE3':0, 'GT3':1, 'T':0, 'A':1, 'teacher': 0, 'health':1,
        'services':2, 'at_home':3, 'other': 4, 'home':1, 'reputation':2, 'course': 3, 'mother':2, 'father':3,
        'no':0, 'yes': 1, 'GP': 0, 'MS': 1}

#portugal math variables with 5% significance from f test
math5 = [1,2,3,4,5,6,10,12,14,15,17,19]
#portugal math variables with 10% significance from f test
math10 = [1,2,3,4,5,6,10,11,12,14,15,16,17,19]
#portugal math variables with 5% significance from f test
por5 = [0,1,2,3,4,5,6,7,8,10,11,12,15,16,17,18,19,20,21,22,23]
#portugal language variables with 10% significance from f test
por10 = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23]
def readIn(path, sigLevel, setName):
    infile = open(path, 'r')
    dataReader = csv.reader(infile, delimiter=',')
    raw = []
    i = 0
    for row in dataReader:
        if i != 0:
            for i in range(0, len(row)):
                if row[i].isdigit():
                    row[i] = int(row[i])
                elif path == middle_east:
                    row[i] = midDict.get(row[i])
                else:
                    row[i] = porDict.get(row[i])
            if path != middle_east and i != 0:
                if row[-1] < 6.66:
                    row[-1] = 0
                elif row[-1] > 6.66 and row[-1] < 13.33:
                    row[-1] = 1
                else:
                    row[-1] = 2
        raw.append(row)
        i += 1
    if sigLevel == 0:
        return raw
    raw = cleanSets(setName, raw, sigLevel)

    return raw

def cleanSets(setName, raw, sigLevel):
    use = []
    data = []
    if setName == "midEast" or sigLevel == 0:
        return raw
    elif setName == "math":
        if sigLevel == 5:
            use = math5
        else:
            use = math10
    else:
        if sigLevel == 5:
            use = por5
        else:
            use = por10
    i = 0
    for row in raw:
        newRow = []
        for num in use:
            newRow.append(row[num])
        data.append(newRow)
    return data


def runSvm(raw):
    labels = raw[0]
    data = []
    target = []

    for row in raw[1:]:
        data.append(row[0:-1])
        target.append(row[-1])

    clf = svm.SVC(kernel = 'rbf')
    data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, target, test_size = 0.33, random_state = random.randrange(0,50) )
    clf.fit(data_train, target_train)
    predicts = clf.predict(data_test)
    acc  = accuracy_score(target_test, predicts)
    return acc

def testSvm(raw):
    total = 0.0
    for i in range(0, 25):
        total += runSvm(raw) * 100
    acc = total/25.0
    return acc

def main(args):
    if len(args) != 2:
        print("please run as follows \n./svm.py <feature significance level> <kernel> \nfeature significance level should be either 0, 5 or 10 (zero will use all of the variables)")
        sys.exit()
    mid = readIn(middle_east, int(args[1]), "midEast")
    math = readIn(portugal_math, int(args[1]), "math")
    por = readIn(portugal_por, int(args[1]), "por")

    #middle east (having trouble with values being too big)
    '''
    print("Starting middle east")
    acc = runSvm(mid)
    print("mid east accuracy is: ", acc)
    '''
    #portugal-math
    print("Starting portugal math")
    acc = testSvm(math)
    print("portugal math accuracy is: ", acc)

    print("Starting portugal language")
    acc = testSvm(por)
    print("portugal language accuracy is: ", acc)

if __name__ == '__main__':
    main(sys.argv)
