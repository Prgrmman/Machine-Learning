#!/usr/bin/python

#returns the feature with its values mapped to 0 or 1
def convertToBinary(feature, itemMappedToZero):
	for index, item in enumerate(feature):
		if item == itemMappedToZero:
			feature[index] = 0
		else:
			feature[index] = 1
	return feature

#returns matrix of sub features
def handleNominal(feature):
    numFeatrues = feature.size
    valueSet = set(list(feature))
    valueList = list(valueSet)
    valueList.sort()
    # create dummy variables. For k possible values, create k-1 values
    dummys = [[0] * numFeatrues for i in range(len(valueList)-1)]

    # go through all values
    for i,value in enumerate(list(feature)):
        valueIndex = valueList.index(value)
        if valueIndex >= len(dummys):
            continue
        dummys[valueIndex][i] = 1
        
    dummys = numpy.array(dummys)
    dummys = dummys.T
    dummys = dummys.astype(numpy.int)
    return dummys

#Features from F-test with significance of 5% math:
#['sex', 'age', 'address', 'Medu', 'Fedu', 'Mjob', 'traveltime', 'failures', 'paid', 'higher', 'romantic', 'goout']

#Features from F-test with significance of 10% math:
#['sex', 'age', 'address', 'Medu', 'Fedu', 'Mjob', 'traveltime', 'studytime', 'failures', 'paid', 'higher', 'internet', 'romantic', 'goout']

#Features from F-test with significance of 5% language:
#['school', 'sex', 'age', 'address', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'traveltime', 'studytime', 'failures', 'higher', 
#'internet', 'romantic', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

#Features from F-test with significance of 10% language:
#['school', 'sex', 'age', 'address', 'Medu', 'Fedu', 'Mjob', 'Fjob', 'reason', 'guardian', 'traveltime', 
#'studytime', 'failures', 'schoolsup', 'higher', 'internet', 'romantic', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

def main(args):

	if(len(args) != 3):
		print("Incorrect number of args\nusage: python3 regressions.py math/language 5/10")
		return

	path = ""
	training_data_indicies = []

	if(args[1] == "math" and args[2] == '5'):
		path = '../data/portugal/student-mat.csv'
		training_data_indicies = [1,2,3,4,5,6,10,12,14,15,17,19]
	elif(args[1] == "math" and args[2] == '10'):
		path = '../data/portugal/student-mat.csv'
		training_data_indicies = [1,2,3,4,5,6,10,11,12,14,15,16,17,19]
	elif(args[1] == "language" and args[2] == '5'):
		path = '../data/portugal/student-por.csv'
		training_data_indicies = [0,1,2,3,4,5,6,7,8,10,11,12,15,16,17,18,19,20,21,22,23]
	elif(args[1] == "language" and args[2] == '10'):
		path = '../data/portugal/student-por.csv'
		training_data_indicies = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,19,20,21,22,23]

	with open(path, 'r') as f:
		reader = csv.reader(f)
		table = list(reader)
		matrix = numpy.array(table)

    #get training data
    #reshape to make every vector Nx1
	school = matrix[1: , 0].reshape(len(matrix)-1,1)
	sex = matrix[1: , 1].reshape(len(matrix)-1,1)
	age = matrix[1: , 2].reshape(len(matrix)-1,1)
	address = matrix[1: , 3].reshape(len(matrix)-1,1)
	medu = matrix[1: , 6].reshape(len(matrix)-1,1)
	fedu = matrix[1: , 7].reshape(len(matrix)-1,1)
	mjob = matrix[1: , 8].reshape(len(matrix)-1,1)
	fjob = matrix[1: , 9].reshape(len(matrix)-1,1)
	reason = matrix[1: , 10].reshape(len(matrix)-1,1)
	guardian = matrix[1: , 11].reshape(len(matrix)-1,1)
	traveltime = matrix[1: , 12].reshape(len(matrix)-1,1)
	studytime = matrix[1: , 13].reshape(len(matrix)-1,1)
	failures = matrix[1: , 14].reshape(len(matrix)-1,1)
	schoolsup = matrix[1: , 15].reshape(len(matrix)-1,1)
	paid = matrix[1: , 17].reshape(len(matrix)-1,1)
	higher = matrix[1: , 20].reshape(len(matrix)-1,1)
	internet = matrix[1: , 21].reshape(len(matrix)-1,1)
	romantic = matrix[1: , 22].reshape(len(matrix)-1,1)
	freetime = matrix[1: , 24].reshape(len(matrix)-1,1)
	goout = matrix[1: , 25].reshape(len(matrix)-1,1)
	dalc = matrix[1: , 26].reshape(len(matrix)-1,1)
	walc = matrix[1: , 27].reshape(len(matrix)-1,1)
	health = matrix[1: , 28].reshape(len(matrix)-1,1)
	absences = matrix[1: , 29].reshape(len(matrix)-1,1)

    #what we are predicting
	g3_score = matrix[1:,32].reshape(len(matrix)-1,1)

    #convert all data to vectors or matricies
	school = convertToBinary(school, 'GP').astype(numpy.int)
	sex = convertToBinary(sex, 'M').astype(numpy.int)
	age = age.astype(numpy.int)
	address = convertToBinary(address, 'U').astype(numpy.int)
	medu = medu.astype(numpy.int)
	fedu = fedu.astype(numpy.int)
	mjob = handleNominal(mjob.T[0])
	fjob = handleNominal(fjob.T[0])
	reason = handleNominal(reason.T[0])
	guardian = convertToBinary(guardian, 'mother').astype(numpy.int)
	traveltime = traveltime.astype(numpy.int)
	studytime = studytime.astype(numpy.int)
	failures = failures.astype(numpy.int)
	schoolsup = convertToBinary(schoolsup, 'no').astype(numpy.int)
	paid = convertToBinary(paid, 'no').astype(numpy.int)
	higher = convertToBinary(higher, 'no').astype(numpy.int)
	internet = convertToBinary(internet, 'no').astype(numpy.int)
	romantic = convertToBinary(romantic, 'no').astype(numpy.int)
	freetime = freetime.astype(numpy.int)
	goout = goout.astype(numpy.int)
	dalc = dalc.astype(numpy.int)
	walc = walc.astype(numpy.int)
	health = health.astype(numpy.int)
	absences = absences.astype(numpy.int)
	g3_score = g3_score.astype(numpy.int)

	features = [school,sex,age,address,medu,fedu,mjob,fjob,reason,guardian,traveltime,studytime,
	failures,schoolsup,paid,higher,internet,romantic,freetime,goout,dalc,walc,health,absences]

	#combine training data into one matrix
	training_data = features[training_data_indicies[0]]
	for i,j in enumerate(training_data_indicies):
		if(i > 0):
			training_data = numpy.hstack((training_data, features[j]))

    #use linear regression model
	model = LinearRegression()

	averagedPrediction = numpy.zeros(math.ceil((0.33*len(training_data))))
	averagedPrediction = averagedPrediction.reshape(len(averagedPrediction),1)
	averagedError = averagedPrediction

	for i in range(1,10):

		#divide into train and test sets randomly, 10 times, and average the predictions
		training_data_train, training_data_test, g3_score_train, g3_score_test = cross_validation.train_test_split(training_data, g3_score, test_size=0.33)
		model.fit(training_data_train, g3_score_train)

		prediction = model.predict(training_data_test)
		error = numpy.abs(numpy.subtract(prediction,g3_score_test))

		averagedPrediction = numpy.add(averagedPrediction,prediction)
		averagedError = numpy.add(averagedError,error)

	for i,j in enumerate(averagedPrediction):
		averagedPrediction[i] = j/10
	for i,j in enumerate(averagedError):
		averagedError[i] = j/10

	plt.hist(averagedError)
	plt.xlabel("prediction error")
	plt.ylabel("number of examples")
	plt.show()

if __name__ == '__main__':
    import csv
    import sys
    import math
    import numpy
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    from sklearn import cross_validation
    import statsmodels.api as sm
    from scipy import stats
    main(sys.argv)
else:
    import numpy
