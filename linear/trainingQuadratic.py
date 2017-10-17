# quadratic classifier
# mean vector and covariance matrix of
# every class are calculated from 
# training dataset

# to ensure always division float
from __future__ import division
from numpy import linalg as LA
import numpy as np
import scipy.io as sio

outFilename  = 'data/quadraticParameters.mat'

# the algorithm begins with a matrix of feature vectors (features) and
# a column vector of labels (labels)

# some numbers for training and labels (for example)
training = np.array([[1,1,4],[2,4,5],[1,4,3],[4,5,1],[2,2,6],[7,2,4],[8,1,4],[5,1,3],[3,2,4],[4,1,4],[6,5,1],[7,2,4],[8,1,4],[3,5,1],[4,2,4],[5,1,4],[6,5,1],[2,4,7],[1,4,4],[5,1,5],[2,4,1],[1,4,3],[5,1,5],[2,4,6],[2,3,7],[3,4,8],[3,1,2]])
#labels = np.array([1,2,3,0,4])
labels = np.array([0,0,0,0,0,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3,4,4,4,4])

#print "training: ", training
# first step extract mean vector and covariance matrix for each class

## split training dataset by classes
# lists
slovak = []
french = []
spanish = []
german = []
polish = []

slo=0
fre=0
spa=0
ger=0
pol=0

for (featureVector, label) in zip(training, labels):
	#print "labels: ", label
	#print "featureVector: ", featureVector
	if label == 0:
		slovak.append(featureVector)
		slo=slo+1
	elif label == 1:
		french.append(featureVector)
		fre=fre+1
	elif label == 2:
		spanish.append(featureVector)
		spa=spa+1
	elif label == 3:
		german.append(featureVector)
		ger=ger+1
	elif label == 4:
		polish.append(featureVector)
		pol=pol+1
	else:
		print "label out of range."

###########
# probabilities each language
total = slo+fre+spa+ger+pol
#print "total: ", total

pslo=slo/total
pfre=fre/total
pspa=spa/total
pger=ger/total
ppol=pol/total

#print "pslo: ", pslo
#print "pfre: ", pfre
#print "pspa: ", pspa
#print "pger: ", pger
#print "ppol: ", ppol

#########
# change type of the variables (list to array) to work with numpy
slovak=np.array(slovak)
french=np.array(french)
spanish=np.array(spanish)
german=np.array(german)
polish=np.array(polish)

#print "languages:"
#print "slovak: \n", slovak
#print "french: \n", french
#print "spanish: \n", spanish
#print "german: \n", german
#print "polish: \n", polish

#########
# mean vector each class
meanSlo = np.mean(slovak, axis=0)
meanFre = np.mean(french, axis=0)
meanSpa = np.mean(spanish, axis=0)
meanGer = np.mean(german, axis=0)
meanPol = np.mean(polish, axis=0)

#print "mean vector:"
#print "slovak: \n", meanSlo
#print "french: \n", meanFre
#print "spanish: \n", meanSpa
#print "german: \n", meanGer
#print "polish: \n", meanPol

#print "mean vector shape:"
#print "slovak: \n", meanSlo.shape
#print "french: \n", meanFre.shape
#print "spanish: \n", meanSpa.shape
#print "german: \n", meanGer.shape
#print "polish: \n", meanPol.shape

############
# covariance matrix each class
covSlo = np.cov(slovak.T)
covFre = np.cov(french.T)
covSpa = np.cov(spanish.T)
covGer = np.cov(german.T)
covPol = np.cov(polish.T)

#print "covariance matrix:"
#print "slovak: \n", covSlo
#print "french: \n", covFre
#print "spanish: \n", covSpa
#print "german: \n", covGer
#print "polish: \n", covPol

#print "covariance matrix shape:"
#print "slovak: \n", covSlo.shape
#print "french: \n", covFre.shape
#print "spanish: \n", covSpa.shape
#print "german: \n", covGer.shape
#print "polish: \n", covPol.shape


sio.savemat(outFilename, {'meanSlo':meanSlo, 'meanFre':meanFre, 'meanSpa':meanSpa, 'meanGer':meanGer, 'meanPol':meanPol, 'covSlo':covSlo, 'covFre':covFre, 'covSpa':covSpa, 'covGer':covGer, 'covPol':covPol, "pslo":pslo, 'pfre':pfre, 'pspa':pspa, 'pger':pger, 'ppol':ppol})

print "Training QDA done."









