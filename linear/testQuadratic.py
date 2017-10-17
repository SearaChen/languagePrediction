from numpy import linalg as LA
import numpy as np
import scipy.io as sio
import csv

# some samples (sintetic)
test = np.array([[1,1,4],[2,4,5],[1,4,3],[4,5,1],[2,2,6],[2,4,1],[1,4,3]])

parameters = sio.loadmat('data/quadraticParameters.mat')

meanSlo = parameters['meanSlo']
meanFre = parameters['meanFre']
meanSpa = parameters['meanSpa']
meanGer = parameters['meanGer']
meanPol = parameters['meanPol']

covSlo = parameters['covSlo']
covFre = parameters['covFre']
covSpa = parameters['covSpa']
covGer = parameters['covGer']
covPol = parameters['covPol']

pslo = parameters['pslo']
pfre = parameters['pfre']
pspa = parameters['pspa']
pger = parameters['pger']
ppol = parameters['ppol']

#print "meanSlo: \n", meanSlo.shape
#print "meanFre: \n", meanFre.shape
#print "meanSpa: \n", meanSpa.shape
#print "meanGer: \n", meanGer.shape
#print "meanPol: \n", meanPol.shape

#print "meanSlo: \n", meanSlo
#print "meanFre: \n", meanFre
#print "meanSpa: \n", meanSpa
#print "meanGer: \n", meanGer
#print "meanPol: \n", meanPol

#############
# log probability each class
logPslo = np.log(pslo)
logPfre = np.log(pfre)
logPspa = np.log(pspa)
logPger = np.log(pger)
logPpol = np.log(ppol)


###
# inverse covariance matrix each class
invCovSlo = LA.inv(covSlo)
invCovFre = LA.inv(covFre)
invCovSpa = LA.inv(covSpa)
invCovGer = LA.inv(covGer)
invCovPol = LA.inv(covPol)

###
ldCovSlo = 0.5*np.log(LA.det(covSlo))
ldCovFre = 0.5*np.log(LA.det(covFre))
ldCovSpa = 0.5*np.log(LA.det(covSpa))
ldCovGer = 0.5*np.log(LA.det(covGer))
ldCovPol = 0.5*np.log(LA.det(covPol))

###
prediction=[]
for sample in test:
	sampleSlo = sample-meanSlo
	sampleFre = sample-meanFre
	sampleSpa = sample-meanSpa
	sampleGer = sample-meanGer
	samplePol = sample-meanPol

	gSlo= logPslo - ldCovSlo -0.5*np.dot(sampleSlo, np.dot(invCovSlo, sampleSlo.T))
	gFre= logPfre - ldCovFre -0.5*np.dot(sampleFre, np.dot(invCovFre, sampleFre.T))
	gSpa= logPspa - ldCovSpa -0.5*np.dot(sampleSpa, np.dot(invCovSpa, sampleSpa.T))
	gGer= logPger - ldCovGer -0.5*np.dot(sampleGer, np.dot(invCovGer, sampleGer.T))
	gPol= logPpol - ldCovPol -0.5*np.dot(samplePol, np.dot(invCovPol, samplePol.T))
	
#	print "gSlo: ", gSlo
#	print "gFre: ", gFre
#	print "gSpa: ", gSpa
#	print "gGer: ", gGer
#	print "gPol: ", gPol
	
	results = np.array([gSlo, gFre, gSpa, gGer, gPol])
	idMax = np.argmax(results)
	#print "idMax: ", idMax
	prediction.append(idMax)

#print "prediction: ", prediction
##
with open('data/predictionQuadratic.csv', 'w') as csvfile:
	temp=csv.writer(csvfile, delimiter=',')	
	temp.writerow(['Id','Category'])
	cont=0
	for label in prediction:
		temp.writerow([cont,label])
		cont+=1
		
print "prediction with QDA done."



