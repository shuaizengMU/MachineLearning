import sys
import os
import math
import numpy
import pprint
import random
import csv

from sklearn import svm
from sklearn import cross_validation
from sklearn import metrics
from sklearn import datasets

from mpl_toolkits.mplot3d.axes3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D

class xyset_class:
	
	def __init__(self):
		self.x = []
		self.y = -1;
	
	def addX(self, x):
		#print(x)
		self.x = x
		
	def addY(self, y):
		self.y = y
	
class train_set_class:
	def __init__(self):
		self.data = []

	def readXYList(self, filename, numLimit):
		with open(filename, newline='') as csvfile:
			spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')

			i = 0
			for row in spamreader:
				i+=1
				if i == 1:
					continue
					
				xyset = xyset_class();
				line = ', '.join(row)
				lineList = [ int(i) for i in line.split(',')]
				
				X = lineList[1:]
				Y = lineList[0]
				
				xyset.addX(X)
				xyset.addY(Y)
				self.data.append(xyset)
				
				if numLimit < i:
					break;
				

	def exportDataToXYList(self):
		xList = []
		yList = []
		for data in self.data:
			xList.append(data.x)
			yList.append(data.y)
		return xList, yList
		
	def printDataSet(self):
		print(self.data[0].x)
		print(self.data[0].y)

		
		

def svmByPackageMachineLearning(xList, yList):
	
	'''
	Example:
	
	iris = datasets.load_iris()
	clf = svm.SVC(kernel='linear', C=1)
	score = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=10)
	print(score)
	print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
	'''
	
	#SVM with kernel
	clf_rbf = svm.SVC(decision_function_shape='ovo', C=10.0, kernel='rbf',  		gamma=5)
	clf_sig = svm.SVC(decision_function_shape='ovo', C=10.0, kernel='sigmoid',  	gamma=5, 		coef0=100.0)
	#clf_pol = svm.SVC(decision_function_shape='ovo', C=10.0, kernel='polynomial', 	gamma=5,		coef0=100.0, degree=4)
	clf_lin = svm.SVC(decision_function_shape='ovo', C=10.0, kernel='linear')

	
	#cross validation 
	score_rbf = cross_validation.cross_val_score(clf_rbf, xList, yList, cv=10)
	score_sig = cross_validation.cross_val_score(clf_sig, xList, yList, cv=10)
	#score_pol = cross_validation.cross_val_score(clf_pol, xList, yList, cv=10)
	score_lin = cross_validation.cross_val_score(clf_lin, xList, yList, cv=10)
	
	
	print("rbf: %0.2f (+/- %0.2f)" % (score_rbf.mean(), score_rbf.std() * 2))
	print("sig: %0.2f (+/- %0.2f)" % (score_sig.mean(), score_sig.std() * 2))
	#print("pol: %0.2f (+/- %0.2f)" % (score_pol.mean(), score_pol.std() * 2))
	print("lin: %0.2f (+/- %0.2f)" % (score_lin.mean(), score_lin.std() * 2))
	
	
def svmByPackageDataMining(xList, yList):
	
	'''
	Example:
	
	iris = datasets.load_iris()
	clf = svm.SVC(kernel='linear', C=1)
	score = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=10)
	print(score)
	print("Accuracy: %0.2f (+/- %0.2f)" % (score.mean(), score.std() * 2))
	'''
	
	#SVM with kernel
	decision_function = 'ovr'
	clf_rbf = svm.SVC(decision_function_shape=decision_function, 	C=100.0, kernel='rbf',  			gamma=1e-7)
	#clf_sig = svm.SVC(decision_function_shape=decision_function, 	C=10.0, kernel='sigmoid',  		gamma=5, 		coef0=100.0)
	#clf_pol = svm.SVC(decision_function_shape=decision_function, 	C=10.0, kernel='polynomial', 	gamma=5,		coef0=100.0, degree=4)
	clf_lin = svm.SVC(decision_function_shape=decision_function, 	C=2.0, kernel='linear')

	#cross validation 
	print("rbf cross validation")
	score_rbf = cross_validation.cross_val_score(clf_rbf, xList, yList, cv=3)
	
	#score_sig = cross_validation.cross_val_score(clf_sig, xList, yList, cv=10)
	#score_pol = cross_validation.cross_val_score(clf_pol, xList, yList, cv=10)
	#print("linear cross validation")
	#score_lin = cross_validation.cross_val_score(clf_lin, xList, yList, cv=10)
	
	#print("rbf: %0.2f (+/- %0.2f)" % (score_rbf.mean(), score_rbf.std() * 2))
	#print("sig: %0.2f (+/- %0.2f)" % (score_sig.mean(), score_sig.std() * 2))
	#print("pol: %0.2f (+/- %0.2f)" % (score_pol.mean(), score_pol.std() * 2))
	#print("lin: %0.2f (+/- %0.2f)" % (score_lin.mean(), score_lin.std() * 2))	
	
	#return score_rbf.mean(), score_lin.mean()
	return score_rbf.mean(), 0

def PCAReduction(xList, componentNum):

	#kpca = KernelPCA(kernel="linear",  n_components=componentNum)
	
	pca = PCA(n_components=componentNum)
	
	X = np.array(xList)
	newXList = pca.fit_transform(X)
	return newXList

	
	

def PCAfunction(vecList, topN):
	
	###2	compute covariance
	#vec = np.random.multivariate_normal([1, 7], [[1, 2], [1, 2]], 10)
	vec = np.array(vecList)
	#vec = np.array(vecList)
	cov_mat_1 = np.cov(vec.T)
	#print(cov_mat_1[100])
	
	###3	compute eigenvals and eigenvecs
	eigen_vals_1, eigen_vecs_1 = np.linalg.eig(cov_mat_1)
	#print(eigen_vals_1[0:50])
	
	
	#4	sort eigenvecs based on descending order of eigenvals
	eigen_pairs_1 = [(np.abs(eigen_vals_1[i]), eigen_vecs_1[:,i]) for i in range(len(eigen_vals_1))]
	#eigen_pairs_1.sort(reverse=True)
	w_1 = np.hstack((eigen_pairs_1[i][1][:, np.newaxis] for i in range(topN)))
	
	#5	perform dimensionality reduction
	X1 = vec.dot(w_1)
	xList = []
	for x in X1:
		tmpList = [ i.real for i in x]
		xList.append(tmpList)
	return xList


def RBF_kernel_PCA(xList, componentNum):
	kpca = KernelPCA(kernel="rbf", fit_inverse_transform=True, gamma=10)
	X_kpca = kpca.fit_transform(xList)
	X_back = kpca.inverse_transform(X_kpca)

	return X_back
	
def rbfParameterSearch(xList, yList):

	decision_function = 'ovr'
	
	#cMaxn = 1000000000001
	#gammaMaxn = 1001
	
	cAns = 0
	gammaAns = 0
	meanMax = 0

	cMaxn = 100001
	gammaMaxn = 100.11
	
	cPara = 0.01
	mapList = []
	while cPara < cMaxn:
		gammaPara = 1e-9
		
		while gammaPara < gammaMaxn:
			clf_rbf = svm.SVC(decision_function_shape=decision_function, 	C=cPara, kernel='rbf',  gamma=gammaPara)
			score_rbf = cross_validation.cross_val_score(clf_rbf, xList, yList, cv=3)
			
			mean = score_rbf.mean()		
			mapList.append([cPara, gammaPara, mean])
			
			if meanMax < mean:
				cAns = cPara
				gammaAns = gammaPara
				meanMax = mean
			
			print(([cPara, gammaPara, mean]))
			gammaPara = gammaPara*10
		cPara = cPara*10
	
	
	print(cAns, gammaAns, meanMax)
	return mapList

def PCAParameterSearch(xList, yList):
	
	scoreMax = 0
	comMax = 0
	category = ""
	
	n_components = 50
	while n_components < 500:
		#newXList = PCAReduction(xList,n_components)
		newXList = PCAfunction(xList,n_components)
		score_rbf, score_lin = svmByPackageDataMining(newXList, yList)		
		if score_rbf > scoreMax:
			scoreMax = score_rbf
			comMax = n_components
			category = "rbf"
		if score_lin > scoreMax:
			scoreMax = score_lin
			comMax = n_components
			category = "linear"
		n_components += 1
		print(len(newXList[0]), score_rbf, score_lin , category)
	print (comMax, scoreMax)
	score_rbf, score_lin = svmByPackageDataMining(xList, yList)	
	print (len(xList[0]), score_rbf,score_lin)

	
##Testing
# load samples and tests
def loaddata(TrainXFileName, TrainYFileName):
	dataXTuple = numpy.load(TrainXFileName)
	dataYTuple = numpy.load(TrainYFileName)

	# testing
	dataXTuple = dataXTuple[0:]
	dataYTuple = dataYTuple[0:].tolist()
	yList = [i for [i] in dataYTuple]
	
	return dataXTuple, yList


def showList(xList, yList):
	#show
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	
	cateList = [[],[],[],[],[],[],[],[],[],[]]
	for i in range(len(yList)):
		y = yList[i]
		x = xList[i]
		cateList[y].append(x)
	
	color = ["red", "black", "yellow", "blue", "green", "pink", "magenta", "cyan", '#eeefff', "#ffefff"]
	
	i = 0
	for list in cateList:
	
		xList = [i[0] for i in list]
		yList = [i[1] for i in list]
		zList = [0 for i in list]

		ax.scatter(xList, yList, zList, color=color[i], marker = '.', alpha=0.8, depthshade=True)
		#plt.scatter(xList, yList, color=color[i], marker = '^', alpha=0.5)
		i+=1
	ax.set_xlabel('X Label')
	ax.set_ylabel('Y Label')
	ax.set_zlabel('Z Label')	
	plt.show()	

def main(argc, argv):
	trainSet = train_set_class()

	filename = "kaggle_train.csv"
	numLimit = 1000000000001
	
	print("Read data.")
	trainSet.readXYList(filename, numLimit)
	
	print("Export data to list.")
	xList, yList = trainSet.exportDataToXYList()
	print(len(yList))
	
	n_components = 77
	print("PCA ", n_components)
	xList = PCAfunction(xList, n_components)
	#PCAParameterSearch(xList, yList)
	#showList(xList, yList)
	
	print("Train.")
	#xList = PCAReduction(xList,50)
	#print(len(xList[0]))
	score_rbf, score_lin = svmByPackageDataMining(xList, yList)
	print(score_rbf, score_lin)
	
	#mapList = rbfParameterSearch(xList, yList)

	#xList, yList =  loaddata("./TrainX.npy", "./TrainY.npy")
	#svmByPackageMachineLearning(xList, yList)

	
	 
		
if __name__ == "__main__":
	main(len(sys.argv), sys.argv)

	
	