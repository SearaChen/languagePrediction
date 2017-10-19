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
import csv


'''
call this module with the method : pre_process
pre_proess takes in argument 'pca'
pca takes on some number <= 0.8, lower number, fewer dimension the data is reduced to 
pca == None, no pca is applied 
'''



def write_to_file(filename,string):
	with open(filename,'a+') as f:
		f.write(string)

def read_in_train():
	train_X=pd.DataFrame.from_csv('train_set_x.csv')
	train_y = pd.DataFrame.from_csv('train_set_y.csv')
	df=  pd.concat([train_X, train_y], axis=1)
	df.columns = ['Text', 'category']
	f = df[df.Text.str.contains("http")== False  ] # removing lines that contain http
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') # removing all numbers, white space, new line and tabs 
	df['Text'] = df['Text'].str.lower()  # converting all symbols to lower case 
	df = df[df['Text'].str.rstrip()!=''] # dropping empty strings
	df.dropna(axis=0, how='any',inplace=True)
	#print df['Text'].hasnans check if all nan correctly dropped
	r = df[['category']].copy()
	r.category = r.category.astype(int)

	text_list = df['Text'].tolist()
	test= pd.notnull(text_list) # no False in the 'test' ... which means all entries are non-null
	tf = TfidfVectorizer(analyzer='char', min_df = 0, smooth_idf=1 , sublinear_tf=False)
	X = tf.fit_transform(text_list).toarray()     # X = Transformed X train value
	df = pd.DataFrame(X, columns=tf.get_feature_names())
	X_train, X_test, y_train, y_test = train_test_split(df,r, test_size=0.2, random_state=1)
	return [X_train, X_test, y_train, y_test]

def read_in_final_test():
	test_X = pd.DataFrame.from_csv('test_set_x.csv')
	return test_X
def apply_PCA(X_train,new_dimension):
	pca = PCA(n_components = new_dimension)
	X_final=pca.fit_transform(X_train)
	return X_final


def write_to_text(l):
	'''
	writing a list to text
	'''
	pass
		


def SVM(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf = svm.NuSVC(kernel='rbf')
	clf.fit(X_train, y_train)	

	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	return accuracy


def random_forest(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf = RandomForestClassifier(max_depth=2, random_state=0)
	clf.fit(X_train, y_train)
	accuracy=clf.score(X_train, y_train)
	return accuracy
 

'''
49 accuracy
'''
def NB(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	#print X_train
	#print y_train
	clf= GaussianNB()
	clf.fit(X_train, y_train)	
	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	print accuracy
	sys.exit()
	#print confusion_matrix(y_test, predict_y)
	return accuracy

def GradientBoosting(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf=GradientBoostingClassifier(learning_rate=0.05)
	clf.fit(X_train, y_train)
	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	predict_y=clf.predict(np.array(X_test))
	print accuracy
	return accuracy 
'''
boosting : 0.51
'''
def AdaBoosting(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	clf=AdaBoostClassifier(learning_rate=0.05) # default being 1 
	clf.fit(X_train,y_train)
	predict_y=clf.predict(np.array(X_test))
	accuracy=clf.score(np.array(X_test), np.array(y_test), sample_weight=None)
	predict_y=clf.predict(np.array(X_test))

	print accuracy
	return accuracy

'''
New model tried:
SVM
Random_forest
NB (just to double check with model)
GradientBoosting
'''

def pre_process(pca):
	X_train, X_test, y_train, y_test=read_in_train()
	if pca != None:
		X_train= apply_PCA(X_train,pca)
		X_test = apply_PCA(y_train, pca)
	else:
		pass
	return X_train, X_test, y_train, y_test
	'''
	make sure the data files are in the same directory, DO NOT CHANGE the name from the one on Kaggle
	'''

def main():
	X_train, X_test, y_train, y_test=read_in_train()
	print len(X_train), len(y_train),len(X_test), len(y_test)
	X_train= apply_PCA(X_train,0.8)
	num_rows, num_cols = X_train.shape
	X_test=apply_PCA(X_test, new_dimension=0.8)
	print X_train.shape
	print X_test.shape

#	accuracy=GradientBoosting(X_train,y_train, X_test, y_test)
	accuracy=NB(X_train,y_train, X_test, y_test)
	print accuracy
if __name__ == '__main__':
	main()
	sys.exit()
	training = np.array([[1,1,4],[1,1,5],[1,1,2],[1,1,3]])
	labels = np.array([[0],[0],[1],[1]])
	labels=np.array([0,0,1,1])
	accuracy=NB(training,labels, [[0,1,2]], [[1]])
	print accuracy 
