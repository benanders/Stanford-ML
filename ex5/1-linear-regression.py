
# Linear regression with bias/variance diagnostics
#
# The task is to predict the amount of water flowing out of a damn, given the
# change in water level over a specific time period.
#
# We're given a training set, cross validation set, and a test set.

import numpy as np
import matplotlib.pyplot as plt
import scipy.io, scipy.optimize

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

# Trains the linear regression model on the given training data
def train(X, y, regularization):
	# Initialise theta to zeros
	initial_theta = np.zeros(X.shape[1])

	# Train parameters to minimise the cost function
	return scipy.optimize.fmin_cg(cost_reg, initial_theta,
		fprime=cost_reg_grad,
		args=(X, y, regularization),
		maxiter=50)

# Calculates the training and validation costs for plotting learning curves
def learning_curves(X, y, X_val, y_val, regularization):
	num_training_examples = X.shape[0]

	# Gradually increase the training set size
	train_costs = np.zeros(num_training_examples)
	val_costs = np.zeros(num_training_examples)
	for i in range(1, num_training_examples + 1):
		# Training subset
		X_train = X[0:i,:]
		y_train = y[0:i]

		# Find the best parameters for this training subset
		theta = train(X_train, y_train, regularization)

		# Calculate the training cost, IGNORING regularization
		train_costs[i - 1] = cost_reg(theta, X_train, y_train, 0.0)

		# Calculate the validation cost, IGNORING regularization
		val_costs[i - 1] = cost_reg(theta, X_val, y_val, 0.0)
	return train_costs, val_costs

# Read input data
data = scipy.io.loadmat("1-data.mat")
X = data["X"]
y = data["y"].flatten()
X_val = data["Xval"]
y_val = data["yval"].flatten()
X_test = data["Xtest"]
y_test = data["ytest"].flatten()

# Plot the training data
plt.subplot(1, 2, 1)
plt.title("Water flowing out of a dam")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
plt.scatter(X, y, s=4)

# Add a column of 1s for the bias parameter to the X and X_val matrices
X = np.append(np.ones((X.shape[0], 1)), X, axis=1)
X_val = np.append(np.ones((X_val.shape[0], 1)), X_val, axis=1)

print("cost ", cost_reg_grad(np.ones(2), X, y, 1.0))

# Calculate the line of best fit using ALL the training data
regularization = 1.0
best_theta = train(X, y, regularization)

# Plot the line of best fit
lx = np.linspace(X[:,1].min(), X[:,1].max(), 2)
lX = np.append(np.ones((lx.size, 1)), lx.reshape((lx.size, 1)), axis=1)
ly = lX.dot(best_theta)
plt.plot(lx, ly, c="r")

# Calculate the training and validation costs vs. number of training examples
train_costs, val_costs = learning_curves(X, y, X_val, y_val, regularization)
print("""
The learning curve shows both the training set and the validation set have
high, roughly equal errors. This means the model has high bias (the linear
model underfits the data).
""")

# Plot training costs and validation costs against number of training
# examples
num_training_examples = X.shape[0]
costs_x = np.arange(1, num_training_examples + 1)
plt.subplot(1, 2, 2)
plt.title("Learning curve")
plt.xlabel("Number of training examples")
plt.ylabel("Cost")
plt.plot(costs_x, train_costs, c="r")
plt.plot(costs_x, val_costs, c="b")
plt.show()
