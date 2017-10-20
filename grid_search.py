import pandas as pd
import re
import sys
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split 
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from sklearn import tree

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
	tf = TfidfVectorizer(analyzer='char', sublinear_tf=False,norm='l2', ngram_range= (1,1))
	tf=tf.fit(text_list)
	X=tf.transform(text_list).toarray()
	X_train = pd.DataFrame(X, columns=tf.get_feature_names())
	X=tf.transform(X_test['Text'].tolist()).toarray()
	X_test = pd.DataFrame(X, columns=tf.get_feature_names())
	print X_train.shape
	print X_test.shape

	return [X_train, X_test, y_train, y_test,tf]



'''
ACCEPT
notes: make sure all kaggle data is in the same directories with no name change !! 

m_values: a list of m_values you want to experiment 
k_values: a list of k_values you want to experiment 

RETURN 
a tuple and a 2d matrix 
'''
def grid_search(m_values,k_values):
	best_score=0
	best_param=[]
	accuracy_matrix=[]
	for m in m_values:
		accuracy_row=[]
		for k in k_values:
			X_train, X_test, y_train, y_test,tf=read_in_train()

			'''
			put how you fit and train your model here
			should have a variable called 'score' that denote the accuracy of the result
			'''
			if score >  best_score:
				best_score= score
				best_param=[m,k]
		accuracy_matrix.append(accuracy_row)
	if best_param == []:
		sys.exit('grid search did not capture ')

	# ------- comment the following section out if no accuracy matrix desired to be graphed ------------- 


	return[best_param]

