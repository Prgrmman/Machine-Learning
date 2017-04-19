#!/usr/bin/python

#TODO
#normalize??

import csv
import numpy
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import statsmodels.api as sm
from scipy import stats

def convertToBinary(feature, itemMappedToZero):
	for index, item in enumerate(feature):
		if item == itemMappedToZero:
			feature[index] = 0
		else:
			feature[index] = 1
	return feature

#Features from F-test with significance of 10%:
#['sex', 'age', 'address', 'Medu', 'Fedu', 'Mjob', 'traveltime', 'studytime', 'failures', 'paid', 'higher', 'internet', 'romantic', 'goout']
portugalPath = '../data/portugal/student-mat.csv'
with open(portugalPath, 'r') as f:
    reader = csv.reader(f)
    table = list(reader)
    matrix = numpy.array(table)

#get training data
#reshape to make every vector Nx1
sex = matrix[1: , 1].reshape(len(matrix)-1,1)
age = matrix[1: , 2].reshape(len(matrix)-1,1)
address = matrix[1: , 3].reshape(len(matrix)-1,1)
medu = matrix[1: , 6].reshape(len(matrix)-1,1)
fedu = matrix[1: , 7].reshape(len(matrix)-1,1)
mjob = matrix[1: , 8].reshape(len(matrix)-1,1)
traveltime = matrix[1: , 12].reshape(len(matrix)-1,1)
studytime = matrix[1: , 13].reshape(len(matrix)-1,1)
failures = matrix[1: , 14].reshape(len(matrix)-1,1)
paid = matrix[1: , 17].reshape(len(matrix)-1,1)
higher = matrix[1: , 20].reshape(len(matrix)-1,1)
internet = matrix[1: , 21].reshape(len(matrix)-1,1)
romantic = matrix[1: , 22].reshape(len(matrix)-1,1)
goout = matrix[1: , 25].reshape(len(matrix)-1,1)

#what we are predicting
g3_score = matrix[1:,32].reshape(len(matrix)-1,1)

#convert data to numeric, binary, or dummy

sex = convertToBinary(sex, 'M').astype(numpy.int)
age = age.astype(numpy.int)

g3_score = g3_score.astype(numpy.int)

#divide into train and test sets randomly
#g3_score_train, g3_score_test, absences_train, absences_test = cross_validation.train_test_split(g3_score, absences, test_size=0.2)


#X is training data, y is associated value
#use linear regression model
#model = LinearRegression()
#model.fit(X, y)
#prediction = model.predict(absences_test)

