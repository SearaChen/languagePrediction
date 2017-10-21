import pandas as pd
import re
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split 
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from nltk.probability import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

'''
call this module with the method : pre_process
pre_proess takes in argument 'pca'
pca takes on some number <= 0.8, lower number, fewer dimension the data is reduced to 
pca == None, no pca is applied 
'''

def generate_test_result(clf,tf,pca):
	df = pd.DataFrame.from_csv('test_set_x.csv')
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') # removing all numbers, white space, new line and tabs 
	df=df.fillna(method='ffill')
	text_list = df['Text'].tolist()
	test_X = tf.transform(text_list).toarray()     # X = Transformed X train value
	print 'before: '+ str(len(tf.get_feature_names()))
	
	if pca == None:
		pass
	else:
		test_X=pca.transform(test_X)

	print test_X.shape
	a=clf.predict(np.array(test_X))
	df= pd.DataFrame(data=a, columns=['Category'])
	df.index.name='Id'
	df.to_csv('multinomialNB_2gram.csv', sep=',')

def read_in_train():
	train_X=pd.DataFrame.from_csv('train_set_x.csv')
	train_y = pd.DataFrame.from_csv('train_set_y.csv')
	df=  pd.concat([train_X, train_y], axis=1)
	df.columns = ['Text', 'category']
	df = df[df.Text.str.contains("http")== False  ] # removing lines that contain http
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') # removing all numbers, white space, new line and tabs 
	df['Text'] = df['Text'].str.lower()  # converting all symbols to lower case 
	r = df[['category']].copy()
	df = df[df['Text'].str.rstrip()!=''] # dropping empty strings, df
	df.dropna(axis=0, how='any',inplace=True) 
	#print df['Text'].hasnans check if all nan correctly dropped
	r = df[['category']].copy()
	r.category = r.category.astype(int)

	# ------------- for own testing -----------------------
	X_train, X_test, y_train, y_test = train_test_split(df,r, test_size=0.2, random_state=1)

	text_list = X_train['Text'].tolist()
	tf = TfidfVectorizer(analyzer='char', sublinear_tf=False,norm='l2', ngram_range= (1,2))
	tf=tf.fit(text_list)
	X=tf.transform(text_list).toarray()
	X_train = pd.DataFrame(X, columns=tf.get_feature_names())
	X=tf.transform(X_test['Text'].tolist()).toarray()
	X_test = pd.DataFrame(X, columns=tf.get_feature_names())
	print X_train.shape
	print X_test.shape

	return [X_train, X_test, y_train, y_test,tf]
	# ------------------------------------------------------
	'''
	#---- this section is for training the entire corpus, for final submission --------
	text_list = df['Text'].tolist()
	test= pd.notnull(text_list) # no False in the 'test' ... which means all entries are non-null
	tf = TfidfVectorizer(analyzer='char', min_df = 0, ngram_range=(1,2), norm='l2')
	tf=tf.fit(text_list)
	X = tf.transform(text_list).toarray()     # X = Transformed X train value
	df = pd.DataFrame(X, columns=tf.get_feature_names())
	X_train, X_test, y_train, y_test = train_test_split(df,r, test_size=0.2, random_state=1)
	return [X_train, X_test, y_train, y_test,tf]
	'''


def apply_PCA_modified (X_train,pca_ratio):
	pca=PCA(n_components=pca_ratio)
	pca=pca.fit(X_train)
	X_final= pca.transform(X_train)
	return (X_final,pca)


def SVM(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf = svm.LinearSVC(kernel='rbf')
	clf.fit(X_train, y_train)	

	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	print accuracy
	return clf


def random_forest(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(X_train, y_train)
	accuracy=clf.score(np.array(X_test), np.array(y_test))
	print accuracy
	return clf
 

def QDL(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf = QuadraticDiscriminantAnalysis(priors=[0.04989,0.51198,0.25267,0.136,0.049])
	clf.fit(X_train, y_train)
	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	print accuracy
	return clf

def NB(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	#print X_train
	#print y_train
	clf= GaussianNB(priors=[0.04989,0.51198,0.25267,0.13646,0.049])
	clf.fit(X_train, y_train)	
	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	print accuracy
	return clf

def GradientBoosting(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf=GradientBoostingClassifier(learning_rate=0.05)
	clf.fit(X_train, y_train)
	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	print accuracy
	return clf
'''
boosting : 0.51
'''
def AdaBoosting(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf=AdaBoostClassifier(learning_rate=0.05) # default being 1 
	clf.fit(X_train,y_train)
	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	print accuracy
	return clf


def logistic_regression(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf=LogisticRegression(C=0.1)
	clf.fit(X_train, y_train)	

	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	print accuracy
	return clf

def multinomial_NB(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf = MultinomialNB(alpha=1)
	#clf = MultinomialNB()
	clf.fit(X_train,y_train)
	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	print accuracy
	return clf




def pre_process(pca_ratio):
	'''
	for teammates' use for pre_processing data
	'''
	X_train, X_test, y_train, y_test,vocab=read_in_train()
	if pca != None:
		X_train,pca= apply_PCA_modified(X_train,pca_ratio)
		X_test = pca.transform(X_test)
	else:
		pass
	return X_train, X_test, y_train, y_test
	'''
	make sure the data files are in the same directory, DO NOT CHANGE the name from the one on Kaggle
	'''
def apply_cross_validation(X_train,y_train, X_test, y_test):
	pass

