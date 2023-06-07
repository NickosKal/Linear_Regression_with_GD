import numpy as np
import pandas as pd


def load_auto():

	# Import data
	auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna()

	# Extract relevant data features
	# x_train = auto[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'year', 'origin']].values
	x_train = auto[['horsepower']]
	y_train = auto[['mpg']]

	# Normalize the data
	x_train_norm = x_train.apply(lambda rec: (rec - rec.mean()) / rec.std(), axis=0)
	y_train_norm = np.array((y_train - y_train.mean()) / y_train.std())

	return x_train_norm, y_train_norm

