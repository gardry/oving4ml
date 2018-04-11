# Adaboost
import numpy as np
from sklearn import tree
import math
import matplotlib.pyplot as plt
import timeit

def readInput(filename): #reads input from file and returns python numpy array
	f=open(filename)
	retArray=[]
	for line in f: 
		retArray.append(line.split(","))
	retArray=retArray[1:]
	for rowI in range(len(retArray)):
		for colI in range(len(retArray[rowI])):
			retArray[rowI][colI]=float(retArray[rowI][colI].strip("\n"))
			if (colI<2):
				retArray[rowI][colI]=int(retArray[rowI][colI])
	return np.array(retArray)

def initializeWeights(trainingDataCount):
	retDist=np.ones(trainingDataCount)
	return np.array(retDist/trainingDataCount)


def classify(training_data,sample_weight): #uses decisionTreeClassifier to classify data
	clf = tree.DecisionTreeClassifier(max_depth=1)
	return clf.fit(training_data[:,2:], training_data[:,1],sample_weight)

class adaBoostTrained():
	def __init__(self,classifiersList,alphaList,maxIter,training_data):
		self.classifiersList=classifiersList
		self.alphaList=alphaList
		self.maxIter=maxIter
		self.training_data=training_data

	def adaPredict(self,testVal):
		retSum=0
		for it in range(self.maxIter):
			retSum+=self.alphaList[it]*self.classifiersList[it].predict(np.array([testVal]))
		if retSum>0:
			return 1
		return -1

	def in_sample_test(self):
		predictions=[]
		for it in range(self.maxIter): #iterate through 
			predictions.append(self.classifiersList[it].predict(self.training_data[:,2:]))
		predictions=np.array(predictions).T
		predictions=np.dot(predictions,self.alphaList)
		predictions[predictions > 0]=1
		predictions[predictions < 0]=-1
		errorArray=np.multiply(predictions,self.training_data[:,1])
		errorArray[errorArray > 0]=0
		errorRate=-sum(errorArray)/self.training_data.shape[0]
		return errorRate

	def out_of_sample_test(self):
		predictions=[]
		for it in range(self.maxIter): #iterate through 
			predictions.append(self.classifiersList[it].predict(self.test_data[:,2:]))
		predictions=np.array(predictions).T
		predictions=np.dot(predictions,self.alphaList)
		predictions[predictions > 0]=1
		predictions[predictions < 0]=-1
		errorArray=np.multiply(predictions,self.test_data[:,1])
		errorArray[errorArray > 0]=0
		errorRate=-sum(errorArray)/self.test_data.shape[0]
		return errorRate

def runAdaBoost(maxIter,training_data):
	training_length=training_data.shape[0]
	weights=initializeWeights(training_length)
	classifiers=[]
	error=[]
	alphas=[]
	for it in range(maxIter):
		#learn classifiers
		currentClassifier=classify(training_data, weights)
		classifiers.append(currentClassifier)

		#count errors
		answers=currentClassifier.predict(training_data[:,2:])
		answers=np.multiply(answers,training_data[:,1])
		answers[answers > 0]=0
		answers=np.multiply(answers,weights)
		current_errors=-sum(answers)
		
		#compute alphas
		currentAlpha=1/2*math.log((1-current_errors)/current_errors)
		alphas.append(currentAlpha)

		#update weight distribution
		exponents=-currentAlpha*np.multiply(training_data[:,1],currentClassifier.predict(training_data[:,2:]))
		weights=np.multiply(weights,np.exp(exponents))

		#normalize weights
		normFac=sum(weights)
		weights=weights/normFac

	return adaBoostTrained(classifiers, alphas, maxIter,training_data)
	#vote from classifiers

def main():
	training_data=readInput("dataset_o4ML\\adaboost_train.csv")
	test_data=readInput("dataset_o4ML\\adaboost_test.csv")
	iterations=[1,2,5,10,20,50,100,200,500,1000]
	training_error=[]
	test_error=[]
	for it in iterations:
		trainedAlgo=runAdaBoost(it,training_data)
		training_error.append(trainedAlgo.in_sample_test())
		trainedAlgo.test_data=test_data
		test_error.append(trainedAlgo.out_of_sample_test())
	output(iterations,training_error,test_error)


def output(x_axis, x_1,x_2):
	N = 5

	ind =np.arange(len(x_axis))  # the x locations for the groups
	width = 0.35       # the width of the bars

	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, x_1, width, color='r')
	rects2 = ax.bar(ind + width, x_2, width, color='y')

	# add some text for labels, title and axes ticks
	ax.set_ylabel('Error rate')
	ax.set_title('Training and test error for different # of iterations')
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(x_axis)

	ax.legend((rects1[0], rects2[0]), ('In sample error', 'Out of sample error'))

	plt.show()