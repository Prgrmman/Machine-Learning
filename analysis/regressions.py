#!/usr/bin/python

#TODO
#normalize??


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
    print(feature)
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
    dummys = dummys.astype(numpy.float)
    return dummys

#Features from F-test with significance of 10%:
#['sex', 'age', 'address', 'Medu', 'Fedu', 'Mjob', 'traveltime', 'studytime', 'failures', 'paid', 'higher', 'internet', 'romantic', 'goout']
def main():
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

    #convert all data to vectors or matricies
    sex = convertToBinary(sex, 'M').astype(numpy.int)
    age = age.astype(numpy.int)
    address = convertToBinary(address, 'U').astype(numpy.int)
    medu = medu.astype(numpy.int)
    fedu = fedu.astype(numpy.int)
    mjob = handleNominal(mjob.T[0])
    traveltime = traveltime.astype(numpy.int)
    studytime = studytime.astype(numpy.int)
    failures = failures.astype(numpy.int)
    paid = convertToBinary(paid, 'no').astype(numpy.int)
    higher = convertToBinary(higher, 'no').astype(numpy.int)
    internet = convertToBinary(internet, 'no').astype(numpy.int)
    romantic = convertToBinary(romantic, 'no').astype(numpy.int)
    goout = goout.astype(numpy.int)
    g3_score = g3_score.astype(numpy.int)

    #combine training data into one matrix
    training_data = sex
    training_data = numpy.hstack((training_data, age))
    training_data = numpy.hstack((training_data, address))
    training_data = numpy.hstack((training_data, medu))
    training_data = numpy.hstack((training_data, fedu))
    training_data = numpy.hstack((training_data, mjob))
    training_data = numpy.hstack((training_data, traveltime))
    training_data = numpy.hstack((training_data, studytime))
    training_data = numpy.hstack((training_data, failures))
    training_data = numpy.hstack((training_data, paid))
    training_data = numpy.hstack((training_data, higher))
    training_data = numpy.hstack((training_data, internet))
    training_data = numpy.hstack((training_data, romantic))
    training_data = numpy.hstack((training_data, goout))

    #divide into train and test sets randomly
    training_data_train, training_data_test, g3_score_train, g3_score_test = cross_validation.train_test_split(training_data, g3_score, test_size=0.2)

    #use linear regression model
    model = LinearRegression()
    model.fit(training_data_train, g3_score_train)
    prediction = model.predict(training_data_test)
    print(prediction)

if __name__ == '__main__':
    import csv
    import numpy
    from sklearn.linear_model import LinearRegression
    from sklearn import cross_validation
    import statsmodels.api as sm
    from scipy import stats
    main()
else:
    import numpy
