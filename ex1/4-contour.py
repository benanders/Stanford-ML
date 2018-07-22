
# Linear Regression
#
# For a number of various cities, the population of the city and the profit a
# hypothetical business makes in that city is given. Our task is to predict how
# much profit the business would make if it were to set up operations in a
# city with a particular population.
#
# This version shows a contour plot of the cost function around the minimum
# parameters found by gradient descent. It plots the path taken by gradient
# descent using a convergent and non-convergent learning rate.

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
	params_history = [np.array((0.0, 0.0))]
	for _ in range(0, iterations):
		next_params = descend(params_history[-1], learning_rate)
		params_history.append(np.copy(next_params))
	return params_history

# Show a contour plot around the minimum parameters
x = np.arange(-10, 10, 0.1)
y = np.arange(-1, 4, 0.1)
xx, yy = np.meshgrid(x, y)
vectorized_cost = np.vectorize(lambda x, y: cost(np.array((x, y))))
z = vectorized_cost(xx, yy)
axes.set_title("Cost Function")
axes.set_xlabel("Feature 0")
axes.set_ylabel("Feature 1")
axes.contour(x, y, z, np.logspace(-2, 3, 20))

# Plot the parameter history on the contour plot for a convergent learning rate
params_history = gradient_descent()
param_x = map(lambda param: param[0], params_history)
param_y = map(lambda param: param[1], params_history)
axes.scatter(param_x, param_y, s=1, c="r")

# Plot the parameter history on the contour plot for a non-convergent learning
# rate
params_history = gradient_descent(0.02501)[:15]
param_x = map(lambda param: param[0], params_history)
param_y = map(lambda param: param[1], params_history)
axes.scatter(param_x, param_y, s=1, c="g")

plt.show()
