
# Linear Regression
#
# For a number of various cities, the population of the city and the profit a
# hypothetical business makes in that city is given. Our task is to predict how
# much profit the business would make if it were to set up operations in a
# city with a particular population.
#
# This version uses basic gradient descent to find the best possible parameters
# that minimise the cost function.

import numpy as np
import matplotlib.pyplot as plt

# Read in the population and profit data from a CSV file
data = np.genfromtxt("1-data.txt", delimiter=",")
populations = data[:,0]
profits = data[:,1]
num_samples = populations.size
num_features = 2

# Create a figure with only 1 set of axes
_, axes = plt.subplots()

# Scatter plot the profit vs. population data
axes.scatter(populations, profits, c="r", s=4)
axes.set_title("Profits from Various Cities")
axes.set_xlabel("Population (in 10,000s)")
axes.set_ylabel("Profit (in $10,000s)")

# Create the features matrix by adding a column of 1s to the population data,
# so we can treat the y intercept of the line as another feature
features = np.stack((np.ones(num_samples), populations), 1)

# Cost function for least squares regression; calculates the difference between
# each of the actual samples and the predicted profit using our model
def cost(parameters):
	predicted = features.dot(parameters)
	return np.sum((predicted - profits) ** 2.0) / (2.0 * num_samples)

# Runs one iteration of gradient descent; subtracts the grad of the cost
# function at `parameters` from each of the parameters
def descend(parameters, learning_rate):
	predicted = features.dot(parameters)
	scale = learning_rate / num_samples
	parameters -= scale * np.transpose(features).dot(predicted - profits)
	return parameters

# Runs `iteration` iterations of gradient descent to minimise the cost function
def gradient_descent(learning_rate = 0.02, iterations = 1500):
	params = np.array((0.0, 0.0))
	for _ in range(0, iterations):
		params = descend(params, learning_rate)
	return params

# Plot the line of best fit using the best parameters
min_params = gradient_descent()
x = np.linspace(populations.min(), populations.max(), num_samples)
y = np.stack((np.ones(num_samples), x), 1).dot(min_params)
axes.plot(x, y, c="b")
plt.show()
