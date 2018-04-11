# k nearest neighbours
import numpy as np
from operator import itemgetter

def readInput(filename): #reads input from file and returns python numpy array
	f=open(filename)
	retArray=[]
	for line in f: 
		retArray.append(line.split(","))
	retArray=retArray[1:]
	for rowI in range(len(retArray)):
		for colI in range(len(retArray[rowI])):
			retArray[rowI][colI]=float(retArray[rowI][colI].strip("\n")) 
	return np.array(retArray)

def distance(point1,point2): #Computes sum-of squares for the distance between two points
	dist=0
	for coordinates in range(len(point1)):
		dist+=(point1[coordinates]-point2[coordinates])**2
	return dist


def runkNN(testPoint,k): #Runs kNN classification algorithm
	# input is point to be classified and k(number of neighbours)
	classes=3
	testPoint=np.array(testPoint)
	trainingData=readInput("dataset_o4ML\knn_classification.csv") #load training data
	distanceLi=[]
	# iterate through training data and compute distances
	for trainingIndex in range(trainingData.shape[0]):  
		distanceLi.append([distance(testPoint,trainingData[trainingIndex,:4]),trainingData[trainingIndex,4]])
		# distanceLi contains lists on form [distance,class]
	distanceLi.sort(key=itemgetter(0)) #sort distances from low to high, class information contained

	classCount=np.zeros(classes) # for n classes, count number of observations
	for relevant in range(k):
		classCount[int(distanceLi[relevant][1])]+=1

	#iterate through list of classes and return the one with most counts
	bestCount=0
	bestInd=None
	for i in range(classes):
		if(classCount[i]>bestCount):
			bestCount=classCount[i]
			bestInd=i
	#return class with most counts among nearest k neighbours
	return bestInd

def runkNNRegression(testPoint,k): #runs kNN regression algorithm
	# input is point to be regressed and k(number of neighbours)
	testPoint=np.array(testPoint)
	trainingData=readInput("dataset_o4ML\knn_regression.csv") #load training data
	distanceLi=[]
	# iterate through training data and compute distances
	for trainingIndex in range(trainingData.shape[0]):
		distanceLi.append([distance(testPoint,trainingData[trainingIndex,:3]),trainingData[trainingIndex,3]])
		# distanceLi contains lists on form [distance,class]
	distanceLi.sort(key=itemgetter(0)) #sort distances from low to high, class information contained

	#compute average value of k nearest neighbors by looping through
	averageVal=0
	for relevant in range(k):
		averageVal+=distanceLi[relevant][1]
	averageVal=averageVal/k
	return averageVal #return average of k nearest

def main():
	#produce output
	print()
	print("Classification:")
	classificationPoint=[6.3,2.7,4.91,1.8]
	print("Point:\t",classificationPoint)
	print("Result:\t",runkNN(classificationPoint,10))
	print()
	print("Regression:")
	regressionPoint=[6.3,2.7,4.91]
	print("Point:\t",regressionPoint)
	print("Result:\t",runkNNRegression(regressionPoint,10))
main()