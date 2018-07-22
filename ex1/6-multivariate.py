
# Multivariate Linear Regression
#
# For a number of various houses, the area of the house, the number of bedrooms,
# and the price the house sold for are given. Our task is to predict how much
# a house with a given area and number of bedrooms would sell for.

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Read in the population and profit data from a CSV file
data = np.genfromtxt("6-data.txt", delimiter=",")
area = data[:,0]
bedrooms = data[:,1]
price = data[:,2]
num_samples = area.size
num_features = 3

# Create a figure with a set of 3D axes
figure = plt.figure()
axes = figure.add_subplot(111, projection="3d")

# Scatter plot the area, bedroom, and price data
axes.scatter(area, bedrooms, price, c="r", s=4)
axes.set_title("House Sales")
axes.set_xlabel("Area")
axes.set_ylabel("Bedrooms")

# Normalize the features
area_mean, area_stddev = np.mean(area), np.std(area)
area_n = (area - area_mean) / area_stddev
bedrooms_mean, bedrooms_stddev = np.mean(bedrooms), np.std(bedrooms)
bedrooms_n = (bedrooms - bedrooms_mean) / bedrooms_stddev

# Create the features matrix
features = np.stack((np.ones(num_samples), area_n, bedrooms_n), 1)

# Cost function for least squares regression; calculates the difference between
# each of the actual samples and the predicted profit using our model
def cost(parameters):
	predicted = features.dot(parameters)
	return np.sum((predicted - price) ** 2.0) / (2.0 * num_samples)

# Runs one iteration of gradient descent; subtracts the grad of the cost
# function at `parameters` from each of the parameters
def descend(parameters, learning_rate):
	predicted = features.dot(parameters)
	scale = learning_rate / num_samples
	parameters -= scale * np.transpose(features).dot(predicted - price)
	return parameters

# Runs `iteration` iterations of gradient descent to minimise the cost function
def gradient_descent(learning_rate = 0.02, iterations = 1500):
	params = np.array((0.0, 0.0, 0.0))
	for _ in range(0, iterations):
		params = descend(params, learning_rate)
	return params

# Plot the line of best fit using the best parameters
min_params = gradient_descent()
x = np.linspace(area.min(), area.max(), num_samples)
x_n = (x - area_mean) / area_stddev
y = np.linspace(bedrooms.min(), bedrooms.max(), num_samples)
y_n = (y - bedrooms_mean) / bedrooms_stddev
z = np.stack((np.ones(num_samples), x_n, y_n), 1).dot(min_params)
axes.plot(x, y, z, c="b")
plt.show()
