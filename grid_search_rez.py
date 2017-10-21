# make sure the read_in.py is in the same directory!!!


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
import read_in

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
			X_train, X_test, y_train, y_test,tf = read_in.read_in_train()

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


	return best_param

