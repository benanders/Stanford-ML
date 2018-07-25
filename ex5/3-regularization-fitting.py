
# Polynomial regression with bias/variance diagnostics
#
# The task is to predict the amount of water flowing out of a damn, given the
# change in water level over a specific time period.
#
# This program plots the training and cross validation errors against a variety
# of regularization parameter values, in order to select the best regularization
# value. It then computes the training set error and accuracy using the best
# regularization value (i.e. the one that achieves the lowest cross validation
# error).

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

# Normalizes a feature
def normalize(X):
	means = np.mean(X, axis=0) # Get column means
	stds = np.std(X, axis=0) # Get column standard deviations
	# Operators apply column-wise to normalize each column
	return (X - means) / stds, means, stds

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

# Use a polynomial of degree 8
X_poly = polynomialize(X, degree)
X_poly_val = polynomialize(X_val, degree)
X_poly_test = polynomialize(X_test, degree)

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

# Compute the training and cross validation costs for a variety of
# regularization values
regularization_values = np.array([0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0,
	3.0, 10.0])
thetas = np.empty((regularization_values.size, degree + 1))
train_costs = np.empty(regularization_values.size)
val_costs = np.empty(regularization_values.size)
for i in range(0, regularization_values.size):
	regularization = regularization_values[i]

	# Train a linear regression model using this regularization value
	# TRAIN HERE WITH REGULARIZATION; this avoids overfitting
	thetas[i,:] = train(X_poly_norm, y, regularization)

	# Save the training and validation set errors
	# DO NOT ADD REGULARIZATION TO THESE VALUES; regularization would distort
	# the returned cost value and not give a true measurement of the algorithm's
	# performance
	train_costs[i] = cost_reg(thetas[i,:], X_poly_norm, y, 0.0)
	val_costs[i] = cost_reg(thetas[i,:], X_poly_val_norm, y_val, 0.0)

# Compute the best test cost
best_idx = np.argmin(val_costs)
print("Best regularization value was " + str(regularization_values[best_idx]))
print("Training cost: " + str(train_costs[best_idx]))
print("Validation cost: " + str(val_costs[best_idx]))

best_theta = thetas[best_idx,:]
test_cost = cost_reg(best_theta, X_poly_test_norm, y_test, regularization_values[best_idx])
print("Test cost: " + str(test_cost))

# Plot the training and validation costs against the regularization parameter
plt.subplot(1, 1, 1)
plt.title("Evaulation of various regularizations")
plt.xlabel("lambda")
plt.ylabel("Cost")
plt.plot(regularization_values, train_costs, c="b")
plt.plot(regularization_values, val_costs, c="g")
plt.show()
