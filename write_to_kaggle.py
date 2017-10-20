import numpy as np
import pandas as pd


'''
accept a list of numbers which indicates your class classification 
'''
def write_to_kaggle(y_pred,directory):
	a=np.array(y_pred)
	df= pd.DataFrame(data=a, columns=['Category'])
	df.index.name='Id'
	df.to_csv(directory, sep=',')
