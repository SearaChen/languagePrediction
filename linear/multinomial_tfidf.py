import itertools
import re
import sys 
import csv
import read_in as rd
import pandas
import operator
import pandas as pd
from math import log


## split training dataset by classes
# lists
reload(sys)
sys.setdefaultencoding('utf8')
sys.maxunicode
prase_data=[]


def class_probability(y_train):
	y_train=y_train.as_matrix().tolist()

	dic={}
	for i in range(5):
		dic[i]=0

	for y in y_train:
		label=y[0]
		dic[label]+=1
	
	total=0
	for i in range(5):
		total+=dic[i]

	class_probability=[]
	for i in range(5):
		class_probability.append(float(dic[i])/total)

	print 'class probability computed!'
	return class_probability


def word_conditional_probability_tfidf(X_train, y_train):

	print 'what is happening?'

	dic={}
	for i in range(5):
		dic[i]={}
	# for i in range(X_train.shape[0]):
	# 	line = X_train.loc[i].tolist()
	# 	label =y_train.get_value(i,0, takeable=True)
	# 	for i in range(0, len(line)):
	# 		try:dic[label][i]+=line[i]
	# 		except: dic[label][i]=line[i]






	# =============== faster version 
	# print 'calculating tfidf weights sum for words'
	# # the position i in the feature vector represents the name of the feature
	X_train=X_train.as_matrix().tolist()
	y_train=y_train.as_matrix().tolist()
	for (line, label) in zip(X_train, y_train):
		label=label[0]
		for i in range(0,len(line)):
			try:dic[label][i]+=line[i]
			except: dic[label][i]=line[i]
		print len(line)
		print len(dic[label])


	print 'knowing vocabulary'
	vocabulary=len(dic[0].keys())
	print 'vocabulary: '+ str(vocabulary)

	print 'calculating mega files tfidf weights'
	# calculating mega document of class j size:
	mega={}
	for category in range(5):
		mega[category]=0
		for label in dic[category].keys():
			mega[category]+=dic[category][label]


	print 'calculating tfidf probility of each word'
	#probability of each word
	for category in range(5):
		for key in dic[category].keys():
			dic[category][key] = log(float(dic[category][key]+1)/ (mega[category]+vocabulary))
			# print dic[category][key]
			# print mega[category]
			# print dic[category][key]
			# sys.exit()

	#------- purely for testing  --------------- 
	per_key=[]
	for key in sorted(dic[0].keys()):
		temp=0
		for category in range(5):
			temp+=dic[category][key]
		per_key.append(temp)

	count =0 
	for i in per_key:
		print i
		count +=1
		if count == 10:
			break
	print 0 in per_key
	print (len(per_key))

	print 'mega dictionary'
	print mega
	# sys.exit()
	# --------------------------------------
	
	print 'all calculations completed!'
	return [dic,mega]



def predict_tfidf(X_test,c_probability,dic,mega,n=1):
	X_test=X_test.as_matrix().tolist()
	result=[]
	for line in X_test:
		# compute 
		prob_temp=[log(c_probability[i]) for i in range(5)]
		for category in range(5):
			for index in range(0, len(line)):

				# if line[index] != 0 :
				#prob_temp[category]=prob_temp[category]-(line[index]*dic[category][index])
				prob_temp[category]=prob_temp[category]+(line[index]*dic[category][index])
				#prob_temp[category]=prob_temp[category]+log(line[index])+dic[category][index]
				#print prob_temp[category], log(line[index]), dic[category][index]
				#sys.exit()

				# else:
				# 	prob_temp[category]=prob_temp[category]-log(mega[category])   
		
		# print 'prob_temp'
		# print prob_temp
		# sys.exit()
		class_index, prob_value = max(enumerate(prob_temp), key=operator.itemgetter(1))
		result.append(class_index)
	return result

def score(y_pred, y_test):
	# y_pred is a list. y_test is a panda fram
	y_test=y_test.as_matrix().tolist()
	correct_num=0
	for i in range(len(y_pred)):
		y_true=int(y_test[i][0])
		if y_pred[i] == y_true:
			correct_num+=1
	accuracy=float(correct_num)/len(y_pred)
	accuracy = float("{0:.2f}".format(accuracy))

	return accuracy

def kaggle_predict(class_prob, word_dic, mega, directory, tf):
	df = pd.DataFrame.from_csv('test_set_x.csv')
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') # removing all numbers, white space, new line and tabs 
	df=df.fillna(method='ffill')
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') 
	df['Text'] = df['Text'].str.replace(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])','')
	text_list = df['Text'].tolist()
	kaggle_list = tf.transform(text_list).toarray()  # X = Transformed X train value
	X_test = pd.DataFrame(kaggle_list, columns=tf.get_feature_names())

	result= predict_tfidf(X_test,class_prob,word_dic, mega)


	df= pd.DataFrame(data=result, columns=['Category'])
	df.index.name='Id'
	df.to_csv(directory, sep=',')



def tifidf_multinomial():
	X_train, X_test, y_train, y_test,tf = rd.read_in_train()
	class_prob = class_probability(y_train)
	print class_prob
	word_dic, mega= word_conditional_probability_tfidf (X_train,y_train)
	result= predict_tfidf(X_test,class_prob,word_dic, mega)
	print result
	accuracy=score(result,y_test)
	print accuracy
	kaggle_predict(class_prob, word_dic, mega, 'NB_hand_tfidf_unigram.csv',tf)
if __name__ == '__main__':
	tifidf_multinomial()






















