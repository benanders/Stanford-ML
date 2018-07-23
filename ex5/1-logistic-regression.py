
# Polynomial regression with bias/variance diagnostics
#
# The task is to predict the amount of water flowing out of a damn, given the
# change in water level over a specific time period.
#
# We're given a training set, cross validation set, and a test set.

import numpy as np
import matplotlib.pyplot as plt
import scipy.io, scipy.optimize

# Read input data
data = scipy.io.loadmat("1-data.mat")
X = data["X"]
y = data["y"].flatten()
X_val = data["Xval"]
y_val = data["yval"].flatten()
X_test = data["Xtest"]
y_test = data["ytest"].flatten()

# Plot the training data
_, axes = plt.subplots()
axes.set_title("Water flowing out of a dam")
axes.set_xlabel("Change in water level")
axes.set_ylabel("Water flowing out of the dam")
axes.scatter(X, y, s=4)

# Regularized logistic regression cost function
def cost_reg(theta, X, y, regularization):
	num_training_examples = X.shape[0]

	# Make a prediction using the current set of parameters
	prediction = X.dot(theta)

	# Calculate the predictive cost
	cost = np.sum((prediction - y) ** 2)

	# Add on the regularization penalty, ignoring the bias weights
	reg_cost = regularization * np.sum(theta[1:] ** 2)
	return (cost + reg_cost) / (2.0 * num_training_examples)

# Regularized logistic regression gradient function
def cost_reg_grad(theta, X, y, regularization):
	num_training_examples = X.shape[0]

	# Make a prediction using the current set of parameters
	prediction = X.dot(theta)

	# Calculate the gradient
	grad = np.transpose(X).dot(prediction - y)

	# Add regularization to the gradient
	grad_reg = regularization * theta
	grad_reg[0] = 0.0 # Ignore regularization for bias parameter
	return (grad + grad_reg) / num_training_examples

# Add a column of 1s for the bias parameter to the X matrix
num_training_examples = X.shape[0]
X = np.append(np.ones((num_training_examples, 1)), X, axis=1)

# Initialise theta to zeros
initial_theta = np.zeros(X.shape[1])

# Train parameters to minimise the cost function
best_theta = scipy.optimize.fmin_cg(cost_reg, initial_theta,
	fprime=cost_reg_grad,
	args=(X, y, 1.0),
	maxiter=50)

# Plot the line of best fit
lx = np.linspace(X[:,1].min(), X[:,1].max(), 2)
lX = np.append(np.ones((lx.size, 1)), lx.reshape((lx.size, 1)), axis=1)
ly = lX.dot(best_theta)
axes.plot(lx, ly, c="r")
plt.show()
