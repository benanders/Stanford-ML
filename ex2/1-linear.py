
# Linear logistic regression classifier
#
# The task is to predict whether to admit a person to a university based on
# their scores in 2 entrance exams using a linear decision boundary. We use
# scipy's optimize library to determine the parameters that minimise the cost
# function (rather than implementing our own gradient descent again).

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

# Read the data in
data = np.genfromtxt("1-data.txt", delimiter=",")
exam1 = data[:,0]
exam2 = data[:,1]
admitted = data[:,2]
num_samples = admitted.size
num_features = 3

# Plot the classification data
_, axes = plt.subplots()
axes.set_title("Admissions")
axes.set_xlabel("Exam 1 Score")
axes.set_ylabel("Exam 2 Score")
axes.scatter(exam1, exam2, c=admitted, marker='x')

# Features matrix, containing the additional column of ones
features = np.stack((np.ones(num_samples), exam1, exam2), 1)

# Sigmoid function
def sigmoid(value):
	return 1.0 / (1.0 + np.exp(-value))

# Cost function for logistic regression
def cost(parameters):
	predicted = sigmoid(features.dot(parameters))
	positive_weights = -admitted * np.log(predicted)
	negative_weights = -(1 - admitted) * np.log(1 - predicted)
	costs = positive_weights + negative_weights
	return np.sum(costs) / num_samples

# Returns an array of `num_features` partial derivatives of the cost function,
# evaluated at the given parameters
def dcost(parameters):
	predicted = sigmoid(features.dot(parameters))
	difference = predicted - admitted
	return np.transpose(features).dot(difference) / num_samples

# Minimise the parameters for the cost function
initial_params = np.array((0, 0, 0))
info = scipy.optimize.minimize(cost, initial_params, jac=dcost)
min_params = info.x

# Graph the decision boundary by analytically solving for its linear equation
#
# If min_params = (theta0, theta1, theta2)
# h = g(theta0 + x * theta1 + y * theta2) = 0.5
# => theta0 + x * theta1 + y * theta2 = 0
# => y = (-x * theta1 - theta0) / theta2
x = np.linspace(exam1.min(), exam1.max(), 2)
y = (-x * min_params[1] - min_params[0]) / min_params[2]
axes.plot(x, y)
plt.show()
