import sys
import os
import math
import numpy
import pprint
import random
import sklearn.metrics

from tempfile import TemporaryFile

fn = open('outY.npy', "r")
L = []
for line in fn:
	L.append(float(line))
fn.close()

#print (len(L))
predY = numpy.array(L).reshape(len(L),1)
#print(a)
#numpy.save('predY', a)

TestY = numpy.load('TestY.npy') 
print(TestY)
Mat = sklearn.metrics.confusion_matrix(TestY,predY) 
score=numpy.sum(numpy.diagonal(Mat))

print(Mat)
print(score)