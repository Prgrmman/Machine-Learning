#!/usr/bin/python
import csv
import numpy
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation
import statsmodels.api as sm
from scipy import stats

portugalPath = '../data/portugal/student-mat.csv'
with open(portugalPath, 'r') as f:
    reader = csv.reader(f)
    table = list(reader)
    matrix = numpy.array(table)

#get g3 score and absences
g3_score = matrix[2:,32]
absences = matrix[2:,29]

g3_score = g3_score.astype(numpy.float)
absences = absences.astype(numpy.float)

g3_score = g3_score.reshape(-1,1)
absences = absences.reshape(-1,1)

#divide into train and test sets randomly
g3_score_train, g3_score_test, absences_train, absences_test = cross_validation.train_test_split(g3_score, absences, test_size=0.2)

#use linear regression model
model = LinearRegression()
model.fit(g3_score_train,absences_train)


# prints summary of of regression
X = sm.add_constant(absences_train)
olsModel = sm.OLS(g3_score_train, X)
est = olsModel.fit()
print(est.summary())



