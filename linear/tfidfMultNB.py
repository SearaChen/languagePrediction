import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
import numpy as np

inputFilename  = 'matricestrainingValidationData.npz'

#########
def save_data(X_train, X_test, y_train, y_test, vocab):
	print "save function ..."
	outFilename  = 'matricestrainingValidationData.npz'
	np.savez(outFilename, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, vocab=vocab)
	#sio.savemat(outFilename, {'X_train':X_train, 'X_test':X_test, 'y_train':y_train, 'y_test':y_test})
	print "save done."
	return

######
# read
def read_in_data():
##
	#read data
	train_X=pd.DataFrame.from_csv('train_set_x.csv')
	train_y= pd.DataFrame.from_csv('train_set_y.csv')
##	
	# data + categories (class)
	df=pd.concat([train_X, train_y], axis=1)
	df.columns = ['Text', 'category']
##	
	#clean data and become lowercase
	df = df[df.Text.str.contains("http")== False  ] # removing lines that contain http
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') # removing all numbers, white space, new line and tabs 
	df['Text'] = df['Text'].str.lower()
	df=df.fillna(method='ffill')
	df = df[df['Text'].str.rstrip()!=''] # dropping empty strings, df
	df.dropna(axis=0, how='any',inplace=True) 
##	
# copy labels
	r = df['category'].tolist()
	#r = df[['category']].copy()
	#r.category = r.category.astype(int)
	#print "r: \n", r
	#split data to training and validation
	X_train, X_test, y_train, y_test = train_test_split(df,r, test_size=0.2, random_state=42)

	# extract features from text (count of characters)
	#text_list = df['Text'].tolist()
	text_train = X_train['Text'].tolist()
	text_test = X_test['Text'].tolist()
	
	#print "data_X: ", text_list
	# count characters TRAIN
	vectorizer = CountVectorizer(analyzer='char')
	#vectorizer_train = CountVectorizer(analyzer='char')
	#vectorizer_test = CountVectorizer(analyzer='char')
	
	vect_train = vectorizer.fit_transform(text_train)
	vocab = vectorizer.get_feature_names()
	#print "voc: \n", vocab
	
	# count characters TEST with features from TRAIN (vocabulry)
	vectorizer = CountVectorizer(analyzer='char', vocabulary=vocab)
	vect_test  = vectorizer.fit_transform(text_test)
	
	X_train = vect_train.toarray()
	X_test = vect_test.toarray()
##	
	
	
	#print X_train
	print X_train.shape
	print X_train[0:10,0:10]
	print len(y_train)
	print y_train[:10]
	
	#print X_test
	print X_test.shape
	print X_test[0:10,0:10]
	print len(y_test)
	print y_test[:10]
	
	
	
	return  [X_train, X_test, y_train, y_test, vocab]

#####################
def weight_cal(X_train, y_train):
	# count of each feature for all classes minus the count of them in class of interest 
	nsamples, nfeatures = X_train.shape
	print "samples, features: ", nsamples, nfeatures
	N=5 #number of classes
	# inicialization 
	num=np.zeros([N,nfeatures])
	den=np.zeros(5)
	weights=np.zeros([N,nfeatures])
	acc_classes=np.zeros([N,nfeatures])
	
	for (sample,label) in zip(X_train, y_train):
		acc_classes[label]+=sample
	
	print "acc:\n ", acc_classes[:,540:]	
	
	#vector of the total count each feature 
	countFeat=np.sum(acc_classes,axis=0)
	print "countFeat: ", countFeat[540:]	
		
	# numerator weights formulation
	for label in range(5):
		num[label]=np.add(countFeat,-acc_classes[label])
	# add 1 to num in order to avoid num cero 
	num=np.add(1,num)
	print "num: \n", num[:,540:]
	
	# denominator weights formulation
	vec_col  = np.sum(acc_classes,axis=1)
	totalAcc = np.sum(vec_col)
	print "vec_col: ", vec_col
	print "total: ", totalAcc
	
	for label in range(5):
		den[label]=nfeatures + totalAcc - vec_col[label]
	print "den: ", den

	# 
	#weights = np.log(np.divide(num,den))
	#print "weight shape: ", weights.shape
		
	return weights


####
# MAIN

print "Start ..."
# this code extract features
#[X_train, X_test, y_train, y_test, vocab] = read_in_data()
#save data disk
#save_data(X_train, X_test, y_train, y_test, vocab)

# load features precalculated
data = np.load(inputFilename)
X_train=data['X_train']
X_test=data['X_test']
y_train=data['y_train']
y_test=data['y_test']
vocab=data['vocab']

#Now weight function calculation for each features (character)
weights = weight_cal(X_train, y_train)




	
	

