from numpy import linalg as LA
import numpy as np
import scipy.io as sio
import csv

# some samples (sintetic)
test = np.array([[1,1,4],[2,4,5],[1,4,3],[4,5,1],[2,2,6],[2,4,1],[1,4,3]])

parameters = sio.loadmat('data/naiveBayesParameters.mat')

meanSlo = parameters['meanSlo']
meanFre = parameters['meanFre']
meanSpa = parameters['meanSpa']
meanGer = parameters['meanGer']
meanPol = parameters['meanPol']

varSlo = parameters['varSlo']
varFre = parameters['varFre']
varSpa = parameters['varSpa']
varGer = parameters['varGer']
varPol = parameters['varPol']

#print "varSlo: ", varSlo

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

##############
# log sum variance each class
lvarSlo = np.sum(np.log(varSlo))
lvarFre = np.sum(np.log(varFre))
lvarSpa = np.sum(np.log(varSpa))
lvarGer = np.sum(np.log(varGer))
lvarPol = np.sum(np.log(varPol))

#print "lvarSlo: ", lvarSlo
#print "lvarFre: ", lvarFre
#print "lvarSpa: ", lvarSpa
#print "lvarGer: ", lvarGer
#print "lvarPol: ", lvarPol

###
prediction=[]
for sample in test:
	sampleSlo = np.sum(((sample-meanSlo)**2)/(varSlo**2))
	sampleFre = np.sum(((sample-meanFre)**2)/(varFre**2))
	sampleSpa = np.sum(((sample-meanSpa)**2)/(varSpa**2))
	sampleGer = np.sum(((sample-meanGer)**2)/(varGer**2))
	samplePol = np.sum(((sample-meanPol)**2)/(varPol**2))
	
	gSlo = logPslo - lvarSlo - 0.5*sampleSlo
	gFre = logPfre - lvarFre - 0.5*sampleFre
	gSpa = logPspa - lvarSpa - 0.5*sampleSpa
	gGer = logPger - lvarGer - 0.5*sampleGer
	gPol = logPpol - lvarPol - 0.5*samplePol
	
	results = np.array([gSlo, gFre, gSpa, gGer, gPol])
	#print "results: ", results
	idMax = np.argmax(results)
	#print "idMax: ", idMax
	prediction.append(idMax)
	
with open('data/predictionNaiveBayes.csv', 'w') as csvfile:
	temp=csv.writer(csvfile, delimiter=',')	
	temp.writerow(['Id','Category'])
	cont=0
	for label in prediction:
		temp.writerow([cont,label])
		cont+=1


print "prediction with Naive Bayes done."




