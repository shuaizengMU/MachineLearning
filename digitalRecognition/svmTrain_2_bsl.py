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
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d.axes3d import get_test_data

from sklearn.decomposition import KernelPCA
from sklearn.decomposition import PCA

from mpl_toolkits.mplot3d import Axes3D


import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure as fig


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
	
	def readXList(self, filename, numLimit):
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
				
				X = lineList
				Y = -1
				
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
	clf_rbf = svm.SVC(decision_function_shape=decision_function, 	C=100.0, kernel='rbf',  			gamma=1e-7*4.5)
	#clf_sig = svm.SVC(decision_function_shape=decision_function, 	C=10.0, kernel='sigmoid',  		gamma=5, 		coef0=100.0)
	#clf_pol = svm.SVC(decision_function_shape=decision_function, 	C=10.0, kernel='polynomial', 	gamma=5,		coef0=100.0, degree=4)
	clf_lin = svm.SVC(decision_function_shape=decision_function, 	C=2.0, kernel='linear')
	print(clf_rbf)
	
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
	#newX = pca.fit_transform(X)
	pca.fit(X)
	newX = pca.transform(X)
	

	newXList = []
	for x in newX:
		tmpList = [ i.real for i in x]
		newXList.append(tmpList)
	return newXList

def PCAReduction_pair(xList, xTestList, componentNum):

	#kpca = KernelPCA(kernel="linear",  n_components=componentNum)
	
	pca = PCA(n_components=componentNum)
	
	X = np.array(xList)
	XTest = np.array(xTestList)
	#newX = pca.fit_transform(X)
	pca.fit(X)
	newX = pca.transform(X)
	newXTest = pca.transform(XTest)
	

	newXList = []
	for x in newX:
		tmpList = [ i.real for i in x]
		newXList.append(tmpList)
		
	newXTestList = []
	for x in newXTest:
		tmpList = [ i.real for i in x]
		newXTestList.append(tmpList)
		
	return newXList, newXTestList

def PCAfunction_single(vecList, topN):
	
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


def PCAfunction_paired(vecList, topN):
	
	###2	compute covariance
	#vec = np.random.multivariate_normal([1, 7], [[1, 2], [1, 2]], 10)
	vec = np.array(vecList)
	#testVec = np.array(vecTestList)
	
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
		
	#6	testing data perform dimensionality reduction
	#X2 = testVec.dot(w_1)
	#testXList = []
	#for x in X2:
	#	tmpList = [ i.real for i in x]
	#	testXList.append(tmpList)

	return xList, w_1

def doPCAwithEignvector(vecList, w_1):
	vec = np.array(vecList)
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
	
	n_components = 53
	while n_components < 61:
		#newXList = PCAReduction(xList,n_components)
		newXList = PCAfunction_single(xList,n_components)
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


def PCAParameterSearch_addFeature(xList, yList, newXListFeatureList):
	
	scoreMax = 0
	comMax = 0
	category = ""
	
	n_components = 55
	while n_components < 56:
		#newXList = PCAReduction(xList,n_components)
		xList = [xList[idx]+newXListFeatureList[idx] for idx in range(len(xList))]
		print(xList[0])
		#newXList = PCAfunction_single(xList,n_components)
		newXList = PCAReduction(xList,n_components)
		
		#newXList = newXListFeatureList
		print(newXList[0], newXList[1])
		
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

def svmTesting(xList, yList, testXList):
	
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
	clf_rbf = svm.SVC(decision_function_shape=decision_function, 	C=100.0, kernel='rbf',  			gamma=1e-7*4)

	#cross validation 
	#print("rbf cross validation")
	#score_rbf = cross_validation.cross_val_score(clf_rbf, xList, yList, cv=3)
	
	#fitting
	print("trainning")
	clf_rbf.fit(xList, yList) 
	
	#predict
	print("predict")
	result = clf_rbf.predict(testXList)
	
	return result
	'''
	acc = 0
	length = len(result)
	for idx in range(length):
		if result[idx] == yList[idx]:
			acc += 1;
	print (acc/length)
	'''
	
	#return score_rbf.mean(), score_lin.mean()
	#return score_rbf.mean(), 0	

#0.982380812226
def addNonZeroNumberFeature(xList, listLen):
	for idx in range(listLen):
		num = 0
		for x in xList[idx]:
			if x != 0:
				num += 1
		newFeature = [[num]]
	return newFeature

#0.982333196577
def addGreatNNumberFeature(xList, N, listLen):
	for idx in range(listLen):
		num = 0
		for x in xList[idx]:
			if x >  N:
				num += 1
		xList[idx].append(num)
	return xList	

def addEnergyFearture(xList, listLen):
	featureList = []
	for idx in range(listLen):
		num = 0
		for x in xList[idx]:
			if x >  0:
				num += int(x/60)
		featureList.append([num])
	return featureList	


def addRowGreatN(xList, listLen, N):
	newFeatureList = []
	for idx in range(listLen):
		tempList = xList[idx]
		newFeature = [0 for i in range(28)]
		num = 0
		for i in range(len(tempList)):
			if tempList[i] > N:
				index = int(i/28)
				newFeature[index] += 1
		newFeature = [ i*20 for i in newFeature]
		newFeatureList.append(newFeature)
	
	return newFeatureList
	


def addRowGreatN_addN(xList, listLen, N):
	newFeatureList = []
	for idx in range(listLen):
		tempList = xList[idx]
		newFeature = [0 for i in range(28)]
		num = 0
		for i in range(len(tempList)):
			if tempList[i] > N:
				index = int(i/28)
				newFeature[index] += 1
				num += 1
		newFeature = [ i*20 for i in newFeature]
		newFeatureList.append(newFeature)
		newFeatureList.append(num)
	
	return newFeatureList



def addRowGreatN_addN_new(xList, listLen, N):
	newFeatureList = []
	for idx in range(listLen):
		tempList = xList[idx]
		newFeature = [0 for i in range(28*2)]
		num = 0
		for i in range(len(tempList)):
			if tempList[i] > N:
				index = int(i/28)
				newFeature[index] += 1
				newFeature[i%28+28] += 1;
				num += 1
		newFeature = [ i*20 for i in newFeature]
		newFeatureList.append(newFeature)
		newFeatureList.append(num)
	
	return newFeatureList



def addColGreatN(xList, listLen, N):
	newFeatureList = []
	for idx in range(listLen):
		tempList = xList[idx]
		newFeature = [0 for i in range(28)]
		num = 0
		for i in range(len(tempList)):
			if tempList[i] > N:
				newFeature[i%28] += 1;
		newFeatureList.append(newFeature)
	return newFeatureList

def addAreaMean(xList, listLen):
	newFeatureList = []
	nAreaLen = 14
	nAreaNum = 28/nAreaLen
	nAreaNum *= nAreaNum
	for idx in range(listLen):
		tempList = xList[idx]
		newFeature = [0 for i in range(nAreaLen*nAreaLen)]
		for i in range(len(tempList)):
			if tempList[i] == 0:
				continue
			
			rowIdx = int(i/28/(28/nAreaLen))
			colIdx = int(i%28/(28/nAreaLen))
			
			#newFeature[rowIdx*nAreaLen + colIdx] += tempList[i];
			newFeature[rowIdx*nAreaLen + colIdx] += 1;
			
		
		#newFeature = [int(value/nAreaNum) for value in newFeature]
		newFeature = [int(value*60) for value in newFeature]
		newFeatureList.append(newFeature)
		#print(newFeatureList)
		#exit(0)
	return newFeatureList	
	
def addFeatureMain(xList, testXList):
	trainLen = len(xList)
	testLen  = len(testXList)

	##add number of non-zero feature as feature 
	newXListFeature = addNonZeroNumberFeature(xList, trainLen)
	newTestXListFeature = addNonZeroNumberFeature(testXList, testLen)
	
	## number of feature that great than N
	#newXListFeature = addGreatNNumberFeature(xList, 10)
	#newTestXListFeature = addGreatNNumberFeature(testXList, 10)
	
	## add energy
	#newXListFeature = addEnergyFearture(xList, trainLen)
	#newTestXListFeature = addEnergyFearture(testXList, testLen)
	
	## add Row
	#newXListFeature = addRowGreatN(xList, trainLen, 0)
	#newTestXListFeature = addRowGreatN(testXList, testLen, 0)
	#
	### add Col
	#newXListFeature = addColGreatN(xList, trainLen, 0)
	#newTestXListFeature = addColGreatN(testXList, testLen, 0)	


	#newXListFeature = addRowGreatN_addN(xList, trainLen, 0)
	#newTestXListFeature = addRowGreatN_addN(testXList, testLen, 0)	
	
	#newXListFeature = addAreaMean(xList, trainLen)
	#newTestXListFeature = addAreaMean(testXList, testLen)	
	
	#print(newXListFeature[0])
	#print(newXListFeature[1])
	
	return newXListFeature, newTestXListFeature
	
	
def outCSV(yList, outFile):
	with open('out.csv', 'w', newline='', encoding='utf8') as f:
		writer = csv.writer(f)
		writer.writerow(['ImageId', 'Label'])
		idx = 1
		for y in yList:
			writer.writerow([idx, y])	
			idx += 1
		#spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])
		#for y in yList:
		#	fp.write(str(y)+"\n")

def drawSVMParaFirgure(xList, yList):
	'''
	#here's our data to plot, all normal Python lists
	x = [1, 2, 3, 4, 5]
	y = [0.1, 0.2, 0.3, 0.4, 0.5]

	intensity = [5, 10, 15, 20, 25,30, 35, 40, 45, 50,55, 60, 65, 70, 75,80, 85, 90, 95, 100,105, .01, 115, 120, 125]

	#setup the 2D grid with Numpy
	x, y = np.meshgrid(x, y)

	#convert intensity (list of lists) to a numpy array for plotting
	intensity = np.array(intensity).reshape((5, 5))
	print(intensity)

	#now just plug the data into pcolormesh, it's that easy!
	plt.pcolormesh(x, y, intensity)
	plt.colorbar() #need a colorbar to show the intensity scale
	plt.show() #boom	
	'''
	gamma = 1e-9*4.5
	gammaList = [ gamma*(10**powE) for powE in range(10)]
	
	C = 0.00001
	CList = [ C*(10**powE) for powE in range(10)]
	
	print(gammaList)
	print(CList)
	
	resultList = []
	scoreList = []
	decision_function = 'ovr'
	for C in CList:
		for gamma in gammaList:
			clf_rbf = svm.SVC(decision_function_shape=decision_function, 	C=C, kernel='rbf',  			gamma=gamma)

			#cross validation 
			#print("rbf cross validation")
			score_rbf = cross_validation.cross_val_score(clf_rbf, xList, yList, cv=3)
			mean = score_rbf.mean()	
			resultList.append({'C':C, 'gamme':gamma, 'score':mean})
			scoreList.append(mean)
	scoreList = np.array(scoreList)
	
	maxScore = 0
	maxResult = []
	for result in resultList:
		score = result['score']
		if maxScore < score:
			maxScore = score
			maxResult = result
	print (maxScore, maxResult)
	
	x = gammaList
	y = CList
	
	x = range(len(gammaList)+1)
	y = range(len(CList)+1)
	intensity = scoreList.reshape((len(gammaList), len(CList)))
	
	#print(x)
	#print(y)
	#print(intensity)
	
	#f1 = scoreList.reshape((6, 6))
	#setup the 2D grid with Numpy
	x, y = np.meshgrid(x, y)

	plt.pcolormesh(x, y, intensity)
	plt.colorbar() #need a colorbar to show the intensity scale
	#plt.show() #boom	
	
	fig = matplotlib.pyplot.gcf()
	fig.set_size_inches(10.5, 8.5)
	fig.savefig("SVM_C_gamma.png", dpi=100)

	
def main(argc, argv):
	trainSet = train_set_class()
	testSet	 = train_set_class()

	filename = "kaggle_train.csv"
	testingFile = "test.csv"
	numLimit = 1000000000001
	
	print("Read data.")
	trainSet.readXYList(filename, numLimit)
	testSet.readXList(testingFile, numLimit)
	
	print("Export data to list.")
	xList, yList = trainSet.exportDataToXYList()
	testXList, _testYList = testSet.exportDataToXYList()
	#print(len(testXList))
	
	pca_search = True
	
	n_components = 55
	
	#add feature
	newXListFeatureList, newTestXListFeatureList = addFeatureMain(xList, testXList)
	
	
	if pca_search == False:
		print("PCA ", n_components)
	
		#single PCA
		#xList, eginVector = PCAfunction_paired(xList, n_components)
		#testXList = doPCAwithEignvector(testXList, eginVector)
		xList = [xList[idx]+newXListFeatureList[idx] for idx in range(len(xList))]
		testXList = [testXList[idx]+newTestXListFeatureList[idx] for idx in range(len(testXList))]
		
		xList, testXList = PCAReduction_pair(xList, testXList, n_components)
		#

		
		
		
		#all PCA 
		#tratinLen = len(xList)
		#testLen = len(testXList)
		#totalList = xList + testXList
		#
		#totalList = PCAfunction_single(totalList, n_components)
		#
		#xList = totalList[0:tratinLen]
		#testXList = totalList[tratinLen:]
		#
		
	
	if pca_search == True:
		#drawSVMParaFirgure(xList, yList)	
		
		#PCAParameterSearch(xList, yList)
		PCAParameterSearch_addFeature(xList, yList, newXListFeatureList)
		#showList(xList, yList)
		
	if pca_search == False:
		print("Train.")
		
		#cross validation
		#score_rbf, score_lin = svmByPackageDataMining(xList, yList)
		#print(score_rbf, score_lin)
		
		##testing
		outFile = "result.txt"
		result = svmTesting(xList, yList, testXList)
		outCSV(result, outFile)
		
		#mapList = rbfParameterSearch(xList, yList)

		#xList, yList =  loaddata("./TrainX.npy", "./TrainY.npy")
		#svmByPackageMachineLearning(xList, yList)

	
	 
		
if __name__ == "__main__":
	main(len(sys.argv), sys.argv)

	
	