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
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn import tree
'''
call this module with the method : pre_process
pre_proess takes in argument 'pca'
pca takes on some number <= 0.8, lower number, fewer dimension the data is reduced to 
pca == None, no pca is applied 
'''
def read_in_split():
	train_X=pd.DataFrame.from_csv('train_set_x.csv')
	train_y = pd.DataFrame.from_csv('train_set_y.csv')
	df=  pd.concat([train_X, train_y], axis=1)
	df.columns = ['Text', 'category']
	df = df[df.Text.str.contains("http")== False  ] # removing lines that contain http
	# removing all numbers, white space, new line and tabs 
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') 
	# remove all emojis 
	df['Text'] = df['Text'].str.replace(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])','')
	df['Text'] = df['Text'].str.lower()  # converting all symbols to lower case 
	
	# removing lines that contains null characters
	df.dropna(axis=0, how='any',inplace=True)

	# seperating X and Y into two dataframes 
	r = df[['category']].copy()
	r.category = r.category.astype(int)
	df=df.drop('category',axis=1)
	# splitting the data
	X_train, X_test, y_train, y_test = train_test_split(df,r, test_size=0.2, random_state=1)

	return [X_train, X_test, y_train, y_test]

def read_in_train():
	train_X=pd.DataFrame.from_csv('train_set_x.csv')
	train_y = pd.DataFrame.from_csv('train_set_y.csv')
	df=  pd.concat([train_X, train_y], axis=1)
	df.columns = ['Text', 'category']
	df = df[df.Text.str.contains("http")== False  ] # removing lines that contain http
	# removing all numbers, white space, new line and tabs 
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') 
	# remove all emojis 
	df['Text'] = df['Text'].str.replace(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])','')
	df['Text'] = df['Text'].str.lower()  # converting all symbols to lower case 
	
	# removing lines that contains null characters
	df.dropna(axis=0, how='any',inplace=True)

	# seperating X and Y into two dataframes 
	r = df[['category']].copy()
	r.category = r.category.astype(int)
	df=df.drop('category',axis=1)
	# splitting the data
	
	# ------------- for own testing -----------------------
	X_train, X_test, y_train, y_test = train_test_split(df,r, test_size=0.2, random_state=1)



	text_list = X_train['Text'].tolist()
	tf = TfidfVectorizer(analyzer='char', sublinear_tf=False,norm='l2', ngram_range= (1,1))
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

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

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
	df.to_csv('logistic_uni_bi_gram_nopca.csv', sep=',')


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
	X_test=np.array(X_test)
	y_test=np.array(y_test)
	clf = MultinomialNB(alpha=1)
	#clf = MultinomialNB()
	clf.fit(X_train,y_train)
	accuracy=clf.score(X_test, y_test, sample_weight=None)

	y_pred=clf.predict(np.array(X_test))
	cnf_matrix = confusion_matrix(y_test, y_pred)
	np.set_printoptions(precision=2)
	class_names=[0,1,2,3,4]
	plt.figure()
	plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

	plt.show()
	
	print accuracy
	return clf

def decision_tree(X_train, y_train, X_test, y_test):
	X_train=np.array(X_train)
	y_train=np.array(y_train)
	X_test=np.array(X_test)
	y_test=np.array(y_test)

	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X_train, y_train)
	accuracy=clf.score(X_test, y_test, sample_weight=None)
	print accuracy
	return clf


def pre_process(pca):
	'''
	for teammates' use for pre_processing data
	'''
	X_train, X_test, y_train, y_test,tf=read_in_train()
	if pca != None:
		X_train,pca= apply_PCA_modified(X_train,pca)
		X_test = pca.transform(X_test)
	else:
		pass
	return X_train, X_test, y_train, y_test
	'''
	make sure the data files are in the same directory, DO NOT CHANGE the name from the one on Kaggle
	'''

def main(pca_ratio):
	X_train, X_test, y_train, y_test,tf=read_in_train()
	print len(X_train), len(y_train),len(X_test), len(y_test)
	X_train,pca= apply_PCA_modified(X_train,pca_ratio)
	X_test= pca.transform(X_test)
	print X_train.shape
	print X_test.shape
	#clf=decision_tree(X_train,y_train, X_test, y_test)
	#generate_test_result(clf,tf,pca=None)

if __name__ == '__main__':
	main(0.8)
	sys.exit()
