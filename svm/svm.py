#!/usr/bin/python3

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn import cross_validation
import csv
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
        'no':0, 'yes': 1}


def readIn(path):
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
                if row[-1]/3 < 6.66:
                    row[-1] = 0
                elif row[-1]/3 > 6.66 and row[-1]/3 < 13.33:
                    row[-1] = 1
                else:
                    row[-1] = 2
        raw.append(row)
        i += 1

    return raw

def runSvm(raw):
    labels = raw[0]
    data = []
    target = []

    for row in raw[1:]:
        data.append(row[0:-1])
        target.append(row[-1])

    print(data)
    clf = svm.SVC(kernel = 'rbf')
    #data_train, data_test, target_train, target_test = cross_validation.train_test_split(data, target, test_size = 0.33)
    clf.fit(data[0:-1], target[0:-1])
    predicts = clf.predict(data_test[-1])
    acc  = accuracy_score(target[-1], predicts)
    return acc

def main():
    mid = readIn(middle_east)
    math = readIn(portugal_math)
    por = readIn(portugal_por)

    #middle east
    print("Starting middle east")
    acc = runSvm(mid)
    print("mid east accuracy is: ", acc)

    #portugal-math
    print("Starting portugal math")
    acc = runSvm(math)
    print("portugal math accuracy is: ", acc)

    print("Starting portugal language")
    acc = runSvm(por)
    print("mid east accuracy is: ", acc)

if __name__ == '__main__':
    main()
