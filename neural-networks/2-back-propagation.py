
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
import numpy.random, numpy.linalg
import scipy.io

# Sigmoid activation function
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# Neural network cost function (without regularization)
def cost(features, params1, params2, correct_digits):
	num_training_examples = features.shape[0]
	num_outputs = params2.shape[0]
	total_cost = 0

	# Sum the cost over each of the training examples
	for i in range(0, num_training_examples):
		# Extract the training example and its correct answer
		training_example = features[i,:]
		correct_digit = correct_digits[i]

		# Create the correct output array, with the correct digit set to 1 and
		# everything else set to 0
		correct_output = np.zeros(num_outputs)
		correct_output[correct_digit] = 1.0

		# Feed forward: compute the output of the neural network with the given
		# params
		layer1_values = training_example
		layer2_values = sigmoid(params1.dot(layer1_values))
		layer2_values = np.append(np.array((1.0)), layer2_values)
		layer3_values = sigmoid(params2.dot(layer2_values))
		output = layer3_values

		# Compare the output with the correct output and calculate the cost
		cost = np.sum(-correct_output * np.log(output) -
			(1.0 - correct_output) * np.log(1.0 - output))
		total_cost += cost
	return total_cost / num_training_examples

regularization_constant = 1.0

# Regularized neural network cost function
def cost_reg(features, params1, params2, correct_digits):
	num_training_examples = features.shape[0]
	non_reg_cost = cost(features, params1, params2, correct_digits)

	# Add on the regularization term, excluding the bias weights (i.e. first
	# column of each matrix)
	reg_cost = np.sum(params1[:,1:] ** 2) + np.sum(params2[:,1:] ** 2)
	return non_reg_cost + reg_cost / (2.0 * num_training_examples)

# Gradient of the sigmoid function
def sigmoid_grad(z):
	sigmoid_z = sigmoid(z)
	return sigmoid_z * (1.0 - sigmoid_z)

# Randomly initializes the weights for the parameter matrices
def rand_init(shape):
	epsilon = 0.12 # range within which to generate values around 0
	return np.random.sample(shape) * (2.0 * epsilon) - epsilon

# Use the back propagation algorithm to compute the partial derivatives of the
# cost function with respect to every weight in every layer of the network
def cost_grad(features, params1, params2, correct_digits):
	# Number of neurons in each layer
	num_training_examples = features.shape[0]
	layer1_num_neurons = features.shape[1] - 1
	layer2_num_neurons = params1.shape[0]
	layer3_num_neurons = params2.shape[0]

	# Used eventually to compute the partial derivatives
	layer1_Delta = np.zeros((layer2_num_neurons, layer1_num_neurons + 1))
	layer2_Delta = np.zeros((layer3_num_neurons, layer2_num_neurons + 1))

	# Sum the gradient of the cost function over each training example
	for i in range(0, num_training_examples):
		# Get the input training example and the correct output vector
		training_example = features[i,:]
		correct_digit = correct_digits[i]
		correct_output = np.zeros(layer3_num_neurons)
		correct_output[correct_digit] = 1.0

		# Forward propagate to get the network's output
		layer1_values = training_example
		layer2_values = sigmoid(params1.dot(layer1_values))
		layer2_values = np.append(np.array((1.0)), layer2_values)
		layer3_values = sigmoid(params2.dot(layer2_values))
		output = layer3_values

		# Calculate the delta values for the output layer
		layer3_delta = output - correct_output

		# Back propagate the delta values to layer 2
		# Note that we don't require the deltas for layer 1
		layer2_delta = np.transpose(params2).dot(layer3_delta) \
			* layer2_values * (1.0 - layer2_values)
		# a * (1.0 - a) is equivalent to the sigmoid gradient. We use this
		# method over calling `sigmoid_grad` to reduce computation

		# Accumulate the gradient
		# We exclude the bias node from layer 2 when calculating the layer 1
		# Delta because there's no connections from any neuron in layer 1 to
		# that bias node, so no opportunity to back-propagate!
		a1 = layer2_delta[1:].reshape((layer2_num_neurons, 1))
		b1 = layer1_values.reshape((1, layer1_num_neurons + 1))
		c1 = a1.dot(b1)
		layer1_Delta += c1

		a2 = layer3_delta.reshape((layer3_num_neurons, 1))
		b2 = layer2_values.reshape((1, layer2_num_neurons + 1))
		c2 = a2.dot(b2)
		layer2_Delta += c2

	params1_grad = layer1_Delta / num_training_examples
	params2_grad = layer2_Delta / num_training_examples

	# Unroll the gradients into a single vector
	grad = np.append(params1_grad, params2_grad)
	return grad

# Returns a gradient vector similar to cost_grad, but computed analytically
# by pertubing each parameter slightly
def cost_grad_numeric(features, params1, params2, correct_digits):
	epsilon = 0.0001

	# Unroll the parameters into a single vector
	params = np.append(params1.reshape((params1.size,)),
		params2.reshape((params2.size,)))

	# Pertube each parameter
	grad = np.zeros(params.size)
	for i in range(0, params.size):
		# Pertube the parameter
		plus = np.copy(params)
		plus[i] += epsilon
		minus = np.copy(params)
		minus[i] -= epsilon

		# Calculate cost and numerically estimate the gradient
		p1 = plus[0:params1.size].reshape(params1.shape)
		p2 = plus[params1.size:].reshape(params2.shape)
		plus_cost = cost(features, p1, p2, correct_digits)
		m1 = minus[0:params1.size].reshape(params1.shape)
		m2 = minus[params1.size:].reshape(params2.shape)
		minus_cost = cost(features, m1, m2, correct_digits)
		grad[i] = (plus_cost - minus_cost) / (2.0 * epsilon)
		print("grad", plus_cost, minus_cost)
	return grad

# Gradient checks our back propagation algorithm using a small test neural
# network with randomly initialized weights
def grad_check():
	# Create a small neural network, since gradient checking is expensive and we
	# don't want the process to take too long
	layer1_num_neurons = 3
	layer2_num_neurons = 5
	layer3_num_neurons = 3
	num_training_examples = 5

	# Randomly initialise the parameters matrices
	params1 = rand_init((layer2_num_neurons, layer1_num_neurons + 1))
	params2 = rand_init((layer3_num_neurons, layer2_num_neurons + 1))

	# Randomly initialise an input vector and output digits vector
	features = rand_init((num_training_examples, layer1_num_neurons))
	features = np.append(np.ones((num_training_examples, 1)), features, axis=1)
	digits = np.mod(np.arange(0, num_training_examples), layer3_num_neurons)

	# Compute the analytical and numerical gradients
	analytic = cost_grad(features, params1, params2, digits)
	numeric = cost_grad_numeric(features, params1, params2, digits)

	# Compare them as two column vectors
	print(np.append(analytic.reshape(analytic.size, 1),
		numeric.reshape(numeric.size, 1), axis=1))

	# Compare their norm
	difference = np.linalg.norm(analytic - numeric) \
		/ np.linalg.norm(analytic + numeric)
	print("Difference: " + str(difference))

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

# Try gradient checking
grad_check()
