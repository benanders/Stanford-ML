
# Non-linear logistic regression classifier
#
# The task is to determine if we should accept or reject a manufatured micro-
# processor, based on its score in two tests. We use a non-linear decision
# boundary (specifically, we include all possible polynomial terms up to the 6th
# order). We again use scipy's optimize library to determine the parameters that
# minimize the cost function. We apply regularization to avoid overfitting.

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

regularization_weight = 1

# Load the data
data = np.genfromtxt("2-data.txt", delimiter=",")
test1 = data[:,0]
test2 = data[:,1]
accepted = data[:,2]
num_samples = accepted.size

# Plot the classification data
_, axes = plt.subplots()
axes.set_title("Microprocessors")
axes.set_xlabel("Test 1 Score")
axes.set_ylabel("Test 2 Score")
axes.scatter(test1, test2, c=accepted, marker='x')

# Maps two input features to a series of polynomial features up to the 6th order
def map_features(feature1, feature2):
	feature1 = np.array(feature1).reshape((-1, 1))
	feature2 = np.array(feature2).reshape((-1, 1))
	degree = 6
	out = np.ones((feature1.shape[0], 1))
	for i in range(1, degree + 1):
		for j in range(0, i + 1):
			out = np.append(out, feature1 ** (i - j) * feature2 ** j, axis=1)
	return out

# Create the feature from the training data
features = map_features(test1, test2)
num_features = features.shape[1] # Count features automatically

# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Cost function for logistic regression
def cost(parameters):
	predicted = sigmoid(features.dot(parameters))
	positive_weights = -accepted * np.log(predicted)
	negative_weights = -(1 - accepted) * np.log(1 - predicted)
	costs = positive_weights + negative_weights
	regularization = regularization_weight * np.sum(parameters[1:] ** 2)
	return (np.sum(costs) + regularization / 2.0) / num_samples

# Returns an array of `num_features` partial derivatives of the cost function,
# evaluated at the given parameters
def dcost(parameters):
	predicted = sigmoid(features.dot(parameters))
	derivative = np.transpose(features).dot(predicted - accepted)
	regularization = regularization_weight * parameters
	regularization[0] = 0.0 # Apply no regularization to the first parameter
	return (derivative + regularization) / num_samples

# Minimise the parameters for the cost function
initial_params = np.zeros(num_features)
info = scipy.optimize.minimize(cost, initial_params, jac=dcost)
min_params = info.x

# Draw the decision boundary by, for each point in a grid, determining what side
# of the decision boundary the point is on; then approximate the decision
# boundary with a contour plot at the transition point on the boundary of the
# grid
x = np.linspace(-1, 1.5, 200)
y = np.linspace(-1, 1.5, 200)
z = np.zeros((x.size, y.size))
for i in range(0, x.size):
	for j in range(0, y.size):
		# Decide which side of the decision boundary the point is on by making a
		# prediction for whether a microprocessor with these test scores would
		# be accepted or rejected
		prediction = sigmoid(map_features(x[i], y[j]).dot(min_params))
		z[i, j] = prediction

# Make a contour plot at the decision boundary, i.e. all the points on the grid
# where we cross from 0 to 1 (i.e. at 0.5)
#
# We need to tranpose z before calling contour for some reason? Not sure why
axes.contour(x, y, np.transpose(z), levels=[0.5], colors="g")
plt.show()
