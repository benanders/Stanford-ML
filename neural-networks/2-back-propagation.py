
# Handwritten digit classification using known weights
#
# We're given the weights of a neural network that predicts which digit a 20x20
# grayscale image represents.
#
# Training data representation: each row of `images` contains the intensity
# value for each pixel in the 20x20 image, flattened into a 400 element array.
# There are 5000 rows of training data.
#
# Neural network architecture: 400 input neurons in the first input layer, 25
# neurons in the second hidden layer, 10 output neurons, one for each digit, in
# the output layer.

import numpy as np
import scipy.io

# Load training data
training_data = scipy.io.loadmat("data-2.mat")
images = training_data["X"]
digits = training_data["y"]
num_samples = images.shape[0]
num_features = images.shape[1]

# To make dealing with MatLab's 1 based indexing easier, the digit 0 is
# represented by 10. Since we're using Python we revert this.
digits = np.array([0 if x == 10 else x for x in digits])

# Add column of 1s for the bias parameter
features = np.append(np.ones((num_samples, 1)), images, axis=1)

# Load pre-trained neural network weights
weights_data = scipy.io.loadmat("weights-2.mat")
params1 = weights_data["Theta1"] # 25x401 weights for input layer
params2	= weights_data["Theta2"] # 10x26 weights for hidden layer

# Because of the annoying 10 = 0 thing above, we need to reorder the weights in
# params2 because the last row (row 10) is for predicing digit 0 (which, in our
# model, should be the first row)
last_row = np.copy(params2[9,:])
other_rows = np.copy(params2[0:9,:])
params2 = np.insert(other_rows, 0, last_row, axis=0)

# Sigmoid activation function
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# Neural network cost function (without regularization)
def cost(params1, params2):
	total_cost = 0

	# Sum the cost over each of the training examples
	for i in range(0, features.shape[0]):
		# Extract the training example and its correct answer
		example = features[i,:]
		correct_digit = digits[i]

		# Create the correct output array, with the correct digit set to 1 and
		# everything else set to 0
		correct_output = np.zeros(10)
		correct_output[correct_digit] = 1.0

		# Feed forward: compute the output of the neural network with the given
		# params
		layer1_values = example
		layer2_values = sigmoid(params1.dot(layer1_values))
		layer2_values = np.append(np.array((1.0)), layer2_values)
		layer3_values = sigmoid(params2.dot(layer2_values))
		output = layer3_values

		# Compare the output with the correct output and calculate the cost
		cost = np.sum(-correct_output * np.log(output) -
			(1.0 - correct_output) * np.log(1.0 - output))
		total_cost += cost
	return total_cost / num_samples

# Calculate the cost with the pre-trained weights
print("Cost: ", cost(params1, params2))
