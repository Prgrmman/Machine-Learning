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



# prints summary of of regression
X = sm.add_constant(absences)
olsModel = sm.OLS(g3_score, X)
est = olsModel.fit()
print(est.summary())



