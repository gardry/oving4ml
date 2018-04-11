# 2.3 (kNN vs SVM vs Random Forest)
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix



def load_data(): # load data from load_digits and split into four parts, training and test and independent and dependent data
	data_raw=load_digits()
	n_data=data_raw.data.shape[0]
	n_training=n_data*8//10 # 80% training data and 20% test data
	n_test=n_data-n_training
	training_x=data_raw.data[:n_training,:]
	training_y=data_raw.target[:n_training]
	test_x=data_raw.data[n_training:(n_training+n_test),:]
	test_y=data_raw.target[n_training:(n_training+n_test)]
	return training_x,training_y,test_x,test_y

class AlgorithmClass: #class storing information about algorithm
	# independent of which algorithm is used. Possible due to genericy among implementations
	def __init__(self,training_x,training_y,test_x,test_y,algorithmName,knn_k=None):
		self.training_x=training_x
		self.training_y=training_y
		self.test_x=test_x
		self.test_y=test_y
		self.algorithmName=algorithmName
		self.knn_k=knn_k #specific parameter for kNN

	def run(self): #instanciates sklearn implementation of algorithms and trains them on training data
		if(self.algorithmName=="KNN"):
			self.algorithm=KNeighborsClassifier(self.knn_k) #default kNN
			self.algorithm.fit(self.training_x,self.training_y)
		elif(self.algorithmName=="SVM"):
			self.algorithm=svm.LinearSVC() #chosen based on validation data results
			self.algorithm.fit(self.training_x, self.training_y)
		else: 
			self.algorithm=RandomForestClassifier() # default RF
			self.algorithm.fit(self.training_x, self.training_y)
		self.error_rate

	def error_rate(self): #computes training and test error
		c=(self.algorithm.predict(self.test_x)==self.test_y)
		self.out_of_sample_error=np.sum(c)/len(c)
		d=(self.algorithm.predict(self.training_x)==self.training_y)
		self.in_sample_error=np.sum(d)/len(d)

	def conf_matrix(self): # Produces output of confusion matrix for given algorithm
		plt.matshow(confusion_matrix(self.test_y,self.algorithm.predict(self.test_x)))
		plt.xlabel("Predicted value")
		plt.ylabel("Actual value value")
		plt.show()

def main(): # Creates all three algorithms, trains them on training data, and produces confusion matrices based on test data
	training_x,training_y,test_x,test_y=load_data()
	algoKNN=AlgorithmClass(training_x,training_y,test_x,test_y,"KNN",knn_k=10)
	algoSVM=AlgorithmClass(training_x,training_y,test_x,test_y,"SVM")
	algoRF=AlgorithmClass(training_x,training_y,test_x,test_y,"RF")
	algoKNN.run()
	algoSVM.run()
	algoRF.run()
	algoKNN.conf_matrix()
	algoSVM.conf_matrix()
	algoRF.conf_matrix()

	