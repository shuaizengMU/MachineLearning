import sys
import os
import math
import numpy
import pprint
import random

from tempfile import TemporaryFile


class model:
	def __init__(self):
		self.a = []
		self.b = 0.0

class GV:
	def __init__(self):
		self.samples = []        
		self.tests = []    
		self.trainingSet = []      
		self.models = []         
		self.diff_dict = []      # 
		self.cur_mno = 0         #
		self.cache_kernel = []   
		self.use_linear = True  
		self.RBF_dlt = 10       

	def init_models(self):
		for i in range(0, gLetterCaseNum):
			m = model()
			for j in range(len(self.samples)):
				m.a.append(0)
			self.models.append(m)

	def init_cache_kernel(self):
		i = 0
		for mi in self.samples: 
			#print ("kernel:",i)
			self.cache_kernel.append([])
			j = 0
			for mj in self.samples:
				if i > j:
					self.cache_kernel[i].append(self.cache_kernel[j][i])
				else:
					self.cache_kernel[i].append(kernel(mi,mj))
				j += 1
			i += 1

class image:
	def __init__(self):
		self.data = []
		self.num = 0
		self.label = []
		self.fn = ""

	def printself(self):
		print ("data")
		for line in self.data:
			print (line)
		print ("num", self.num)
		print ("label", self.label[gv.cur_mno])
		print ("fn", self.fn )

# global variables
gv = GV()
gLetterCaseNum = 8

def parse_image(path):
	img_map = []
	fp = open(path, "r") 
	for line in fp:
		line = line[:-2]
		img_map.append(line)
	return img_map

# load samples and tests
# def loaddata(dirpath, col):
# 	files = os.listdir(dirpath)
# 	for file in files:
# 		img = image()
# 		img.data = parse_image(dirpath + file)
# 		img.num = int(file[0])
# 		img.fn = file
# 		col.append(img)

def kernel(mj, mi):
	if gv.use_linear == True:
		return kernel_linear(mj,mi)
	else:
		return kernel_RBF(mj,mi)

######
# Gaussian kernel
######
def kernel_RBF(mj, mi):

	dlt = gv.RBF_dlt
	ret = 0.0
	for i in range(len(mj.data)):
		ret += math.pow(mj.data[i]- mi.data[i], 2)
	ret = math.exp(-ret/(2*dlt*dlt))
	return ret

######
# linear kernel
######
def kernel_linear(mj, mi):
	ret = 0.0
	for i in range(len(mj.data)):
		ret += mj.data[i] * mi.data[i]
	return ret


# g(x)
def predict(m):
	pred = 0.0
	for j in range(len(gv.samples)):
		if gv.models[gv.cur_mno].a[j] != 0:
			pred += gv.models[gv.cur_mno].a[j] * gv.samples[j].label[gv.cur_mno] * kernel(gv.samples[j],m)
	pred += gv.models[gv.cur_mno].b 
	return pred

# the same as predict(m), only with different parmaters
def predict_train(i):
	pred = 0.0
	for j in range(len(gv.samples)):
		if gv.models[gv.cur_mno].a[j] != 0:
			pred += gv.models[gv.cur_mno].a[j] * gv.samples[j].label[gv.cur_mno] * gv.cache_kernel[j][i]
	pred += gv.models[gv.cur_mno].b 
	return pred

# 
def predict_diff_real(i):

	diff = predict_train(i)
	diff -= gv.samples[i].label[gv.cur_mno]
	return diff

# 
def predict_diff_real_optimized(idx, i, new_ai, j, new_aj, new_b):
	diff = (new_ai - gv.models[gv.cur_mno].a[i])* gv.samples[i].label[gv.cur_mno] * gv.cache_kernel[i][idx]
	diff += (new_aj - gv.models[gv.cur_mno].a[j])* gv.samples[j].label[gv.cur_mno] * gv.cache_kernel[j][idx]
	diff += new_b - gv.models[gv.cur_mno].b
	diff += gv.diff_dict[idx]
	return diff


def init_predict_diff_real_dict():
	gv.diff_dict = []
	for i in range(len(gv.samples)):
		gv.diff_dict.append(predict_diff_real(i))

def update_diff_dict(i, new_ai, j, new_bj, new_b):
	for idx in range(len(gv.samples)):
		# 
		# gv.diff_dict[idx] = predict_diff_real(idx)
		# 
		gv.diff_dict[idx] = predict_diff_real_optimized(idx, i, new_ai, j, new_bj, new_b)

def update_samples_label(num):
	for img in gv.samples:
		if img.num == num:
			img.label.append(1)
		else:
			img.label.append(-1)

######
######
def SVM_SMO_train(T, times, C, Mno, step):
	time = 0
	gv.cur_mno = Mno

	update_samples_label(Mno)

	
	init_predict_diff_real_dict()


	updated = True
	while time < times and updated:
		updated = False
		time += 1
		for i in range(len(gv.samples)):
			ai = gv.models[gv.cur_mno].a[i]
			Ei = gv.diff_dict[i]

			# agaist the KKT

			if (gv.samples[i].label[gv.cur_mno] * Ei < -T and ai < C) or (gv.samples[i].label[gv.cur_mno] * Ei > T and ai > 0):
				for j in range(len(gv.samples)):
					if j == i: 
					#	print("j==i")
						continue



					kii = gv.cache_kernel[i][i]
					kjj = gv.cache_kernel[j][j]
					kji = kij = gv.cache_kernel[i][j] 
					eta = kii + kjj - 2 * kij 

					if eta <= 0: 
					#	print ("eta<0")
						continue
					new_aj = gv.models[gv.cur_mno].a[j] + gv.samples[j].label[gv.cur_mno] * (gv.diff_dict[i] - gv.diff_dict[j]) / eta # f 7.106
					L = 0.0
					H = 0.0
					a1_old = gv.models[gv.cur_mno].a[i]
					a2_old = gv.models[gv.cur_mno].a[j]
					if gv.samples[i].label[gv.cur_mno] == gv.samples[j].label[gv.cur_mno]:
						L = max(0, a2_old + a1_old - C)
						H = min(C, a2_old + a1_old)
					else:
						L = max(0, a2_old - a1_old)
						H = min(C, C + a2_old - a1_old)
					if new_aj > H:
						new_aj = H
					if new_aj < L:
						new_aj = L

					# print (gv.cur_mno)
					# print (gv.samples[i].label[gv.cur_mno] ,gv.samples[j].label[gv.cur_mno])
					# print (a2_old ,a1_old, step)
					if abs(a2_old - new_aj) < step:
					#	print ("j = %d, is not moving enough" % j)
						continue

					new_ai = a1_old + gv.samples[i].label[gv.cur_mno] * gv.samples[j].label[gv.cur_mno] * (a2_old - new_aj) # f 7.109 
					new_b1 = gv.models[gv.cur_mno].b - gv.diff_dict[i] - gv.samples[i].label[gv.cur_mno] * kii * (new_ai - a1_old) - gv.samples[j].label[gv.cur_mno] * kji * (new_aj - a2_old) # f7.115
					new_b2 = gv.models[gv.cur_mno].b - gv.diff_dict[j] - gv.samples[i].label[gv.cur_mno]*kji*(new_ai - a1_old) - gv.samples[j].label[gv.cur_mno]*kjj*(new_aj-a2_old)    # f7.116
					if new_ai > 0 and new_ai < C: new_b = new_b1
					elif new_aj > 0 and new_aj < C: new_b = new_b2
					else: new_b = (new_b1 + new_b2) / 2.0

					update_diff_dict(i, new_ai, j, new_aj, new_b)
					gv.models[gv.cur_mno].a[i] = new_ai
					gv.models[gv.cur_mno].a[j] = new_aj
					gv.models[gv.cur_mno].b = new_b
					updated = True
					#print ("iterate: %d, changepair: i: %d, j:%d" %(time, i, j))
					#break


def test():
	testY = []
	for img in gv.tests: 
		letterIdx = -1
		for mno in range(gLetterCaseNum):
			gv.cur_mno = mno
			if predict(img) > 0:
				letterIdx = mno
				break
		testY.append(letterIdx)
	return testY




def save_models():
	for i in range(gLetterCaseNum):
		fn = open("models_cross/" + str(i) + "_a.model", "w")
		for ai in gv.models[i].a:
			fn.write(str(ai))
			fn.write('\n')
		fn.close()

		fn = open("models_cross/" + str(i) + "_b.model", "w")
		fn.write(str(gv.models[i].b))
		fn.close()



def load_models():
	for i in range(gLetterCaseNum):
		fn = open("models_cross/" + str(i) + "_a.model", "r")
		j = 0
		for line in fn:
			gv.models[i].a[j] = float(line)
			j += 1
		fn.close()
		fn = open("models_cross/" + str(i) + "_b.model", "r")
		gv.models[i].b = float(fn.readline())
		fn.close()



# load samples and tests
def loaddata(col, TrainX, TrainY):
	dataXTuple = numpy.load(TrainX)
	dataYTuple = numpy.load(TrainY)

	# testing
	dataXTuple = dataXTuple[0:]
	dataYTuple = dataYTuple[0:].tolist()

	for i in range(len(dataXTuple)):
		img = image()
		img.data = dataXTuple[i]
		img.num = dataYTuple[i][0]-1
		img.fn = dataYTuple[i][0] - 1
		col.append(img)

def loadTestingData(Col, TrainX):
	dataXTuple = numpy.load(TrainX)

	# testing
	dataXTuple = dataXTuple[0:]

	for i in range(len(dataXTuple)):
		img = image()
		img.data = dataXTuple[i]
		Col.append(img)	

def getInputArgv(targetSign, argvList, nFollowArg, defaultArg=''):
	try:
		nLenArgv = len(argvList)
		for idx in range(nLenArgv):
			if argvList[idx].lower() == targetSign.lower() and nLenArgv > idx+nFollowArg:
				return  argvList[idx+nFollowArg]
		return defaultArg

	except Exception as e:
		print ("Error in checkInputArgv")
		print (e)

		fp = open('debug.log', 'a')
		fp.write("Error in checkInputArgv.\n")
		fp.write(str(e))
		fp.close()



def main(argv):

	testingX 			= getInputArgv('-x', argv,  1, None)
	output 				= getInputArgv('-o', argv,  1, None)
	C 					= getInputArgv('-c', argv,  1, None)
	gv.RBF_dlt			= getInputArgv('-dlt', argv,1, None)
	optUseLinearKernel 	= getInputArgv('-k', argv,  1, None)

	if testingX==None or C==None or gv.RBF_dlt==None or optUseLinearKernel==None:
		print("Usage: py -x testingX -o output_testing_Y -c C -dlt RBFDelta -k useLinearKernel")
		return

	C = float(C)
	gv.RBF_dlt = float(gv.RBF_dlt)

	if optUseLinearKernel == '0':
		gv.use_linear = False
	else:
		gv.use_linear = True

	T = 0.0001
	step = 0.1

	#load trainning data
	loaddata(gv.samples, 'TrainX.npy', 'TrainY.npy')
	loadTestingData(gv.tests, testingX)

	#calculate kernel variable
	gv.init_cache_kernel()

	#set vertor 'a' as [], and 'b' as 0
	gv.init_models()

	#save models for testing
	load_models()
	for i in range(gLetterCaseNum):
		update_samples_label(i)


	testY = test()

	newTestY = []
	for y in testY:
		if y > -1:
			y += 1
		newTestY.append(y)
	#print(newTestY)

	fn = open(output, "w")
	for y in newTestY:
		fn.write(str(y)+"\n")
	fn.close()


	outfile = TemporaryFile()
	yArray= np.asarray(newTestY)
	np.save(outfile, yArray)




if __name__ == '__main__':
	main(sys.argv)


