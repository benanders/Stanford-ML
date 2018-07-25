
# Polynomial regression with bias/variance diagnostics
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
	train_costs = np.zeros(num_training_examples)
	val_costs = np.zeros(num_training_examples)

	# Gradually increase the training set size
	for i in range(1, num_training_examples + 1):
		# Average the training and validation cost over 50 random samples of i
		# elements from the training set
		for _ in range(0, 50):
			# Contains the indices of the i training examples we should use
			example_idxs = np.random.choice(num_training_examples, i)

			# Training subset
			X_train = np.empty((i, X.shape[1]))
			y_train = np.empty(i)
			for j, idx in enumerate(example_idxs):
				X_train[j,:] = X[idx,:]
				y_train[j] = y[idx]

			# Find the best parameters for this training subset
			theta = train(X_train, y_train, regularization)

			# Calculate the training cost, IGNORING regularization
			train_costs[i - 1] += cost_reg(theta, X_train, y_train, 0.0)

			# Calculate the validation cost, IGNORING regularization
			val_costs[i - 1] += cost_reg(theta, X_val, y_val, 0.0)
	return train_costs / 50, val_costs / 50

# The degree of polynomial to fit to the data
degree = 8

# Read input data
data = scipy.io.loadmat("1-data.mat")
X = data["X"]
y = data["y"].flatten()
X_val = data["Xval"]
y_val = data["yval"].flatten()
X_test = data["Xtest"]
y_test = data["ytest"].flatten()

# Generates a new features matrix containing all powers of X up to `degree`.
# DOESN'T add a column of 1s to the start, since we plan on normalizing the
# features after this, and the column of 1s would get turned into 0s by this
# process
def polynomialize(X, degree):
	X = X.reshape((X.size, 1))
	result = np.copy(X)
	for power in range(2, degree + 1):
		result = np.append(result, X ** power, axis=1)
	return result

# Use a polynomial of degree 8
X_poly = polynomialize(X, degree)
X_poly_val = polynomialize(X_val, degree)
X_poly_test = polynomialize(X_test, degree)

# Normalizes a feature
def normalize(X):
	means = np.mean(X, axis=0) # Get column means
	stds = np.std(X, axis=0) # Get column standard deviations
	# Operators apply column-wise to normalize each column
	return (X - means) / stds, means, stds

# Feature normalization, since we're using high polynomial powers of features;
# allows learning to converge much faster
X_poly_norm, X_means, X_stds = normalize(X_poly)

# Normalize validation and test sets using the same mean and stddev as the
# training set
X_poly_val_norm = (X_poly_val - X_means) / X_stds
X_poly_test_norm = (X_poly_test - X_means) / X_stds

# Add column of 1s to each of the training, validation, and test sets
X_poly_norm = np.append(np.ones((X_poly_norm.shape[0], 1)), X_poly_norm, axis=1)
X_poly_val_norm = np.append(np.ones((X_poly_val_norm.shape[0], 1)), X_poly_val_norm, axis=1)
X_poly_test_norm = np.append(np.ones((X_poly_test_norm.shape[0], 1)), X_poly_test_norm, axis=1)

# Calculate the line of best fit using ALL the training data
regularization = 1.0
best_theta = train(X_poly_norm, y, regularization)

# Plot the training data
plt.subplot(1, 2, 1)
plt.title("Water flowing out of a dam")
plt.xlabel("Change in water level")
plt.ylabel("Water flowing out of the dam")
plt.scatter(X, y, s=4)

# Plots a regression line over the given x range
def plot_regression(theta, x, means, stds):
	x_poly = polynomialize(x, degree)
	x_poly_norm = (x_poly - means) / stds # Normalize
	x_poly_norm = np.append(np.ones((x_poly_norm.shape[0], 1)), x_poly_norm, axis=1)
	y = x_poly_norm.dot(theta)
	plt.plot(x, y, c="r")

# Plot the line of best fit on the same figure
x_range = np.linspace(X.min() - 15, X.max() + 15, 100)
plot_regression(best_theta, x_range, X_means, X_stds)

# Calculate the training and validation costs vs. number of training examples
train_costs, val_costs = learning_curves(X_poly_norm, y, X_poly_val_norm, y_val,
	regularization)
print("""
The learning curve shows both the training set and validation set have low
errors, close to 0. This means the model fits well.

Try changing the `regularization` parameter in the script to something other
than the default of 1.0. For 0.0, we see a high validation error and low
training error, indicating high variance (overfitting). For 100.0, we see high
validation and training error, indicating high bias (underfitting).
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
