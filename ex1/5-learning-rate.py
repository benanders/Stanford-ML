
# Linear Regression
#
# For a number of various cities, the population of the city and the profit a
# hypothetical business makes in that city is given. Our task is to predict how
# much profit the business would make if it were to set up operations in a
# city with a particular population.
#
# This plots various learning rates that both cause gradient descent to converge
# and not converge to a stable value.

import numpy as np
import matplotlib.pyplot as plt

# Read in the population and profit data from a CSV file
data = np.genfromtxt("1-data.txt", delimiter=",")
populations = data[:,0]
profits = data[:,1]
num_samples = populations.size
num_features = 2

# Create a figure with 2 sets of axes
_, (convergent, non_convergent) = plt.subplots(1, 2)

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
def gradient_descent(learning_rate=0.02, iterations=1500):
	params_history = [np.array((0.0, 0.0))]
	for _ in range(0, iterations - 1):
		next_params = descend(np.copy(params_history[-1]), learning_rate)
		params_history.append(next_params)
	return params_history

# Plots cost function vs. iterations for a series of learning rates
vectorized_cost = np.vectorize(lambda x, y: cost(np.array((x, y))))
def plot_cost_function(axes, learning_rates, iterations, clip_min=None,
		clip_max=None):
	axes.set_title("Cost vs. Iterations")
	axes.set_xlabel("Iterations")
	axes.set_ylabel("Cost")
	for rate in learning_rates:
		iteration_counts = np.linspace(0, iterations, iterations)
		params_history = gradient_descent(rate, iterations)
		param_x = map(lambda p: p[0], params_history)
		param_y = map(lambda p: p[1], params_history)
		costs = vectorized_cost(param_x, param_y)
		if clip_min and clip_max:
			costs = np.clip(costs, clip_min, clip_max)
		axes.plot(iteration_counts, costs)

# Plot cost vs. iterations for a series of convergent and non-convergent
# learning rates
convergent_rates = np.linspace(0.001, 0.02, 5)
non_convergent_rates = np.linspace(0.02, 0.026, 5)
plot_cost_function(convergent, convergent_rates, 1500, -2, 7)
plot_cost_function(non_convergent, non_convergent_rates, 8)
plt.show()
