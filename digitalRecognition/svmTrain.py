import os
import sys
import csv



def readXY(filename):
	with open(filename, newline='') as csvfile:
		spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
		i = 0
		Y = []
		X = []
		for row in spamreader:
			i+=1
			if i == 1:
				continue
			line = ', '.join(row)
			#lineList = [ int(i) for i in line.split(',')]

			print(line.split(','))
			break

if __name__ == "__main__":
	filename = "kaggle_train.csv"
	readXY(filename)
