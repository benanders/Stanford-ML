
# Multiclass logistic regression classifier
#
# The task is to take a subset of images from the MNIST dataset and classify
# them as one of the digits from 0 to 9. The images are 20x20 pixel grayscale
# images, with intensity values represented as floats between 0 and 1. All
# images have been normalized to a constant background and text color.

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import scipy.io, scipy.optimize

# `images` is a 5000x400 matrix; each row is a flattened 20x20 grayscale image
# representing a single training example.
#
# `digits` is a 5000 element vector of digits 0 to 9 representing the correct
# answer for each training example.
data = scipy.io.loadmat("data-3.mat")
images = data["X"]
digits = data["y"]
num_samples = digits.size
num_features = images.shape[1] + 1 # Add one for bias parameter

# To make dealing with MatLab's 1 based indexing easier, the digit 0 is
# represented by 10. Since we're using Python we revert this.
digits = np.array([0 if x == 10 else x for x in digits])

# We use a simple LINEAR decision boundary; this is akin to a plane in 401
# dimensional space separating each class
#
# Add a column of 1s to the start of the images matrix for the bias parameter
features = np.append(np.ones((num_samples, 1)), images, axis=1)

# Create a new image of size 10*20 x 10*20 by randomly picking 100 rows from the
# training set and joining them all up
examples = np.empty((200, 200))
for row in range(0, 200, 20):
	for column in range(0, 200, 20):
		inverted_image = images[numpy.random.randint(0, 5000)].reshape(20, 20)
		image = np.transpose(inverted_image)
		examples[row:(row+20),column:(column+20)] = image

# Display a 10x10 grid of example images from the dataset
plt.imshow(examples, interpolation="nearest")
# plt.show()

# Sigmoid activation function
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# Logistic regression cost function. `actual` are the "correct" tags for the
# training data, for the particular classifier that we're currently training.
def cost(parameters, actual):
	predicted = sigmoid(features.dot(parameters))
	costs = -actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted)
	return np.sum(costs) / num_samples

# Cost function with regularization
def cost_reg(parameters, actual, reg_weight):
	predicted = sigmoid(features.dot(parameters))
	costs = -actual * np.log(predicted) - (1 - actual) * np.log(1 - predicted)
	reg = (reg_weight / 2.0) * np.sum(parameters[1:] ** 2)
	return (np.sum(costs) + reg) / num_samples

# Returns all partial derivatives of the cost function evaluated at the given
# point
def dcost(parameters, actual):
	predicted = sigmoid(features.dot(parameters))
	return np.transpose(features).dot(predicted - actual) / num_samples

# Partial derivatives of regularized cost function
def dcost_reg(parameters, actual, reg_weight):
	predicted = sigmoid(features.dot(parameters))
	reg = reg_weight * parameters
	reg[0] = 0.0 # Don't regularize the bias parameter
	return (np.transpose(features).dot(predicted - actual) + reg) / num_samples

# Train 10 logistic classifiers, one for each digit. `min_params` stores the
# minimum parameters for each of the 10 classifiers, in each per row
num_classifiers = 10 # One classifier for each digit
classifiers = np.zeros((num_classifiers, num_features))
for digit in range(0, 10):
	print("Training classifier for digit " + str(digit))

	# Create a logical array containing 0s and 1s for each training example; a
	# 1 means the training example is an image of the digit `digit`
	actual = (digits == digit).astype(int)

	# Train the classifier by minimising the regularized cost function
	reg_weight = 1
	min_params = np.zeros(num_features)
	min_params = scipy.optimize.minimize(cost_reg, min_params, jac=dcost_reg,
		args=(actual, reg_weight)).x

	# Each row in the classifiers matrix is the minimum parameters for a
	# classifier for that digit
	classifiers[digit,:] = min_params

# Apply the classifier to each of the images in the training set and measure the
# accuracy
correct = 0
for row in range(0, num_samples):
	image = np.transpose(features[row,:])
	prediction = classifiers.dot(image)

	# The index containing the highest predicted value represents the digit we
	# think this image is
	predicted_digit = np.argmax(prediction)
	if predicted_digit == digits[row]:
		correct += 1
print("Accuracy: " + str(float(correct) / float(num_samples) * 100.0) + "%")
