import itertools
import re
import sys 
import csv
import read_in as rd
import pandas
import operator
import pandas as pd


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

def word_conditional_probability_discrete(X_train, y_train,n):

	X_train=X_train.as_matrix().tolist()
	y_train=y_train.as_matrix().tolist()

	dic={}
	for i in range(5):
		dic[i]={}

	for (line, label) in zip(X_train, y_train):
		label=label[0]
		line=line[0]

		for i in range(0,len(line)-n+1):
			try:dic[label][line[i:i+n]]+=1
			except: dic[label][line[i:i+n]]=1
	

	# calculating size of vocabulary 
	vocabulary=[]
	for category in range(5):
		vocabulary.extend(dic[category].keys()) 
	vocabulary= len ( list(set(vocabulary)))


	# calculating mega document of class j size:
	mega={}
	for category in range(5):
		mega[category]=0
		for label in dic[category].keys():
			mega[category]+=dic[category][label]


	# probability of each word
	for category in range(5):
		for key in dic[category].keys():
			dic[category][key] = float(dic[category][key]+1)/ (mega[category]+vocabulary)

	return [dic,mega]



def predict(X_test,c_probability,dic,mega,n=1):
	X_test=X_test.as_matrix().tolist()
	result=[]
	for line in X_test:
		line=line[0]

		# compute 
		prob_temp=[c_probability[i] for i in range(5)]
		for category in range(5):
			for index in range(0, len(line)-n+1):
				try: 
					prob_temp[category]=prob_temp[category]*dic[category][line[index:index+n]]
				except: prob_temp[category]=prob_temp[category]/float(mega[category])

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
def kaggle_predict(class_prob, word_dic, mega,n, directory):
	df = pd.DataFrame.from_csv('test_set_x.csv')
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') # removing all numbers, white space, new line and tabs 
	df=df.fillna(method='ffill')
	df['Text'] = df['Text'].str.replace(r'[0-9\s\t\n]+', '') 
	# remove all emojis 
	df['Text'] = df['Text'].str.replace(u'([\u2600-\u27BF])|([\uD83C][\uDF00-\uDFFF])|([\uD83D][\uDC00-\uDE4F])|([\uD83D][\uDE80-\uDEFF])','')
	kaggle= df[['Text']].copy()

	result= predict(kaggle,class_prob,word_dic, mega,1)


	df= pd.DataFrame(data=result, columns=['Category'])
	df.index.name='Id'
	df.to_csv(directory, sep=',')


def discrete_multinomial(ngram=1):
	X_train, X_test, y_train, y_test=rd.read_in_split()
	class_prob = class_probability(y_train)
	word_dic, mega= word_conditional_probability_discrete(X_train,y_train,ngram)
	result= predict(X_test,class_prob,word_dic, mega,ngram)
	accuracy= score(result,y_test)
	print accuracy
	kaggle_predict(class_prob, word_dic, mega, ngram, 'NB_hand_bi.csv')

if __name__ == '__main__':
	discrete_multinomial()























