import pandas as pd
import numpy as np

def get_data():
	
	df = pd.read_csv('goal_data.csv')
	
	X, Y = [], []
	for i in set(df.match_no.values):

		x_i = np.array(df.loc[df['match_no'] == i, 
				['home_score', 'away_score', 'time']])

		y_i = np.array(df.loc[df['match_no'] == i, 'winners'].unique())

		X.append(x_i)
		Y.append(y_i)


	return X, Y
