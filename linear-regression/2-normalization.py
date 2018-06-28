
# Linear Regression
#
# For a number of various cities, the population of the city and the profit a
# hypothetical business makes in that city is given. Our task is to predict how
# much profit the business would make if it were to set up operations in a
# city with a particular population.
#
# This version demonstrates feature normalization.

import numpy as np
import matplotlib.pyplot as plt

# Read in the population and profit data from a CSV file
data = np.genfromtxt("./data-1.txt", delimiter=",")
populations = data[:,0]
profits = data[:,1]
num_samples = populations.size
num_features = 2

# Normalize the population data
populations_mean, populations_stddev = np.mean(populations), np.std(populations)
populations_n = (populations - populations_mean) / populations_stddev

# Create a figure with only 1 set of axes
_, axes = plt.subplots()

# Scatter plot the profit vs. population data
axes.scatter(populations, profits, c="r", s=4)
axes.set_title("Profits from Various Cities")
axes.set_xlabel("Population (in 10,000s)")
axes.set_ylabel("Profit (in $10,000s)")

# Create the features matrix by adding a column of 1s to the population data,
# so we can treat the y intercept of the line as another feature
features = np.stack((np.ones(num_samples), populations_n), 1)

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

# Calculate the line of best fit
min_params = gradient_descent()
x_n = np.linspace(populations_n.min(), populations_n.max(), num_samples)
y = np.stack((np.ones(num_samples), x_n), 1).dot(min_params)

# Reverse the normalization on the x axis
x = x_n * populations_stddev + populations_mean

# Plot the line
axes.plot(x, y, c="b")
plt.show()
