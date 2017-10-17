from numpy import linalg as LA
import numpy as np
import scipy.io as sio
import csv

# some samples (sintetic)
test = np.array([[1,1,4],[2,4,5],[1,4,3],[4,5,1],[2,2,6],[2,4,1],[1,4,3]])

parameters = sio.loadmat('data/LDAParameters.mat')

# mean vectors each class
meanSlo = parameters['meanSlo']
meanFre = parameters['meanFre']
meanSpa = parameters['meanSpa']
meanGer = parameters['meanGer']
meanPol = parameters['meanPol']

# covariance matrix for all classes
covTotal = parameters['covTotal']

pslo = parameters['pslo']
pfre = parameters['pfre']
pspa = parameters['pspa']
pger = parameters['pger']
ppol = parameters['ppol']

#############
# log probability each class
logPslo = np.log(pslo)
logPfre = np.log(pfre)
logPspa = np.log(pspa)
logPger = np.log(pger)
logPpol = np.log(ppol)


###
# inverse covariance matrix
invCovTotal = LA.inv(covTotal)

###
# (mean vector) * (inverse covariance matrix) * (mean vector transpose) for each class
meanCovSlo = np.dot(meanSlo, np.dot(invCovTotal,meanSlo.T))
meanCovFre = np.dot(meanFre, np.dot(invCovTotal,meanFre.T))
meanCovSpa = np.dot(meanSpa, np.dot(invCovTotal,meanSpa.T))
meanCovGer = np.dot(meanGer, np.dot(invCovTotal,meanGer.T))
meanCovPol = np.dot(meanPol, np.dot(invCovTotal,meanPol.T))

#print "meanCovSlo: ", meanCovSlo

###
prediction=[]
for sample in test:
	sampleSlo = np.dot(sample, np.dot(invCovTotal,meanSlo.T))
	sampleFre = np.dot(sample, np.dot(invCovTotal,meanFre.T))
	sampleSpa = np.dot(sample, np.dot(invCovTotal,meanSpa.T))
	sampleGer = np.dot(sample, np.dot(invCovTotal,meanGer.T))
	samplePol = np.dot(sample, np.dot(invCovTotal,meanPol.T))

	#print "sampleSlo: ", sampleSlo
	
	gSlo= logPslo - 0.5*meanCovSlo + sampleSlo
	gFre= logPfre - 0.5*meanCovFre + sampleFre
	gSpa= logPspa - 0.5*meanCovSpa + sampleSpa
	gGer= logPger - 0.5*meanCovGer + sampleGer
	gPol= logPpol - 0.5*meanCovPol + samplePol
	
	#print "gSlo: ", gSlo

	results = np.array([gSlo, gFre, gSpa, gGer, gPol])
	idMax = np.argmax(results)
	#print "idMax: ", idMax
	prediction.append(idMax)

#print "prediction: ", prediction
##
with open('data/predictionLDA.csv', 'w') as csvfile:
	temp=csv.writer(csvfile, delimiter=',')	
	temp.writerow(['Id','Category'])
	cont=0
	for label in prediction:
		temp.writerow([cont,label])
		cont+=1
		
print "prediction with LDA done."




