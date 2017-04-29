import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.optimizers import RMSprop
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

def goal_match(x_i):

	x = np.zeros((10, 2))
	for j, k in enumerate(x_i[:,2]//10):
		k = int(k)
		x[k] = x_i[j, :2]
		
	x = np.maximum.accumulate(x)

	return x	


def get_data():
	
	df = pd.read_csv('goal_data.csv')
	n_g = df.match_no.value_counts().max()

	X, Y = [], []
	for ind, i in enumerate(set(df.match_no.values)):

		x_i = np.array(df.loc[df['match_no'] == i, 
				['home_score', 'away_score', 'time']])

		x = goal_match(x_i)

		# 0 - away_team, 1 - draw, 2 - home_team
		y_i = 1 + int(df.loc[df['match_no'] == i, 'winners'].unique()[0])
		
		y = np.zeros(3)
		y[y_i] += 1

		X.append(x)
		Y.append(y)

	return np.array(X), np.array(Y)


def plot_match(goals):

	x = goal_match(goals)

	p = model.predict(np.array([x]))[0]

	plt.plot(np.linspace(0, 90, 10), p[:, 0], label='Away Team Win')
	plt.plot(np.linspace(0, 90, 10), p[:, 1], label='Draw')
	plt.plot(np.linspace(0, 90, 10), p[:, 2], label='Home Team Win')

	score_text = "Score-line\n"
	for g in goals:
		h = g[0]
		a = g[1]
		d = g[2]
		
		s = "{} - {} @ {}mins \n".format(h, a, d)
		score_text += s

	plt.ylim((0, 1))
	plt.xlim((0, 90))
	plt.text(5, 0.9, score_text) 

	plt.ylabel('Prob.')
	plt.xlabel('Time')
	plt.legend()
	plt.tight_layout()
	plt.show()


def acc_at_t(model, X_test, Y_test, t):

	preds = np.argmax(model.predict(X_test)[:, t], axis=1)
	ans = np.argmax(Y_test[:, 0], axis=1)

	correct = np.sum(np.equal(preds, ans))
	return correct / len(Y_test)


print("Importing data...")
X, Y = get_data()

n_seq = 10
n_fet = 2
batch = 64
n_out = Y.shape[1]
split = 100

Y_rep = np.repeat(Y, n_seq, axis=0).reshape(len(Y), n_seq, n_out)
X_train, X_test = X[:split*batch], X[split*batch:]
Y_train, Y_test = Y_rep[:batch*split], Y_rep[split*batch:]

print('Build model...')
model = Sequential()
model.add(LSTM(8, input_shape=(n_seq, n_fet), return_sequences=True))
model.add(TimeDistributed(Dense(n_out, activation='softmax')))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch, epochs=5)
sc = model.evaluate(X_test, Y_test, batch_size=batch)

print("\nModel {0:}: {1:.2f} \nModel {2:}: {3:.2f}".format(model.metrics_names[0], sc[0], model.metrics_names[1], sc[1]))

plot_match(np.array([[1, 0, 20], [1, 1, 50]]))


