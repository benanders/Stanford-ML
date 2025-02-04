
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
import scipy.io, scipy.optimize
import matplotlib.pyplot as plt

# Sigmoid activation function
def sigmoid(z):
	return 1.0 / (1.0 + np.exp(-z))

# Neural network cost function (without regularization)
def cost(params, layer1_size, layer2_size, layer3_size, X, y, regularization):
	num_training_examples = X.shape[0]
	total_cost = 0

	# Unpack the parameter matrices
	params1_shape = (layer2_size, layer1_size + 1)
	params1_size = params1_shape[0] * params1_shape[1]
	params2_shape = (layer3_size, layer2_size + 1)
	params2_size = params2_shape[0] * params2_shape[1]
	params1 = params[0:params1_size].reshape(params1_shape)
	params2 = params[params1_size:].reshape(params2_shape)

	# Sum the cost over each of the training examples
	for i in range(0, num_training_examples):
		# Extract the training example and its correct answer
		training_example = X[i,:]
		correct_digit = y[i]

		# Create the correct output array, with the correct digit set to 1 and
		# everything else set to 0
		correct_output = np.zeros(layer3_size)
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

# Regularized neural network cost function
def cost_reg(params, layer1_size, layer2_size, layer3_size, X, y,
		regularization):
	num_training_examples = X.shape[0]
	non_reg_cost = cost(params, layer1_size, layer2_size, layer3_size, X, y,
		regularization)

	# Unpack the parameter matrices
	params1_shape = (layer2_size, layer1_size + 1)
	params1_size = params1_shape[0] * params1_shape[1]
	params2_shape = (layer3_size, layer2_size + 1)
	params2_size = params2_shape[0] * params2_shape[1]
	params1 = params[0:params1_size].reshape(params1_shape)
	params2 = params[params1_size:].reshape(params2_shape)

	# Add on the regularization term, excluding the bias weights (i.e. first
	# column of each matrix)
	reg_cost = np.sum(params1[:,1:] ** 2) + np.sum(params2[:,1:] ** 2)
	factor = regularization / (2.0 * num_training_examples)
	return non_reg_cost + reg_cost * factor

# Randomly initializes the weights for the parameter matrices
def rand_init(shape):
	epsilon = 0.12 # range within which to generate values around 0
	return np.random.sample(shape) * (2.0 * epsilon) - epsilon

# Use the back propagation algorithm to compute the partial derivatives of the
# cost function with respect to every weight in every layer of the network
def cost_grad(params, layer1_size, layer2_size, layer3_size, X, y,
		regularization):
	# Number of neurons in each layer
	num_training_examples = X.shape[0]

	# Unpack the parameter matrices
	params1_shape = (layer2_size, layer1_size + 1)
	params1_size = params1_shape[0] * params1_shape[1]
	params2_shape = (layer3_size, layer2_size + 1)
	params2_size = params2_shape[0] * params2_shape[1]
	params1 = params[0:params1_size].reshape(params1_shape)
	params2 = params[params1_size:].reshape(params2_shape)

	# Used eventually to compute the partial derivatives
	# Include the weights extending from the bias nodes
	layer1_Delta = np.zeros((layer2_size, layer1_size + 1))
	layer2_Delta = np.zeros((layer3_size, layer2_size + 1))

	# Sum the gradient of the cost function over each training example
	for i in range(0, num_training_examples):
		# Get the input training example and the correct output vector
		training_example = X[i,:]
		correct_digit = y[i]
		correct_output = np.zeros(layer3_size)
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
		a1 = layer2_delta[1:].reshape((layer2_size, 1))
		b1 = layer1_values.reshape((1, layer1_size + 1))
		c1 = a1.dot(b1)
		layer1_Delta += c1

		a2 = layer3_delta.reshape((layer3_size, 1))
		b2 = layer2_values.reshape((1, layer2_size + 1))
		c2 = a2.dot(b2)
		layer2_Delta += c2

	params1_grad = layer1_Delta / num_training_examples
	params2_grad = layer2_Delta / num_training_examples

	# Unroll the gradients into a single vector
	return np.append(params1_grad, params2_grad)

# Gradient of the cost function with regularization
def cost_grad_reg(params, input_size, hidden_size, output_size, X, y,
		regularization):
	num_training_examples = X.shape[0]
	grad = cost_grad(params, input_size, hidden_size, output_size, X, y,
		regularization)

	# Shapes of the parameter matrices for the first (input) and second (hidden)
	# layers
	params1_shape = (hidden_size, input_size + 1)
	params1_size = params1_shape[0] * params1_shape[1]
	params2_shape = (output_size, hidden_size + 1)
	params2_size = params2_shape[0] * params2_shape[1]

	# Unpack the gradient matrices again
	params1 = params[0:params1_size].reshape(params1_shape)
	params2 = params[params1_size:].reshape(params2_shape)
	params1_grad = grad[0:params1_size].reshape(params1_shape)
	params2_grad = grad[params1_size:].reshape(params2_shape)

	# Add the regularization term, excluding the bias weights
	factor = regularization / num_training_examples
	params1_grad[:,1:] += factor * params1[:,1:]
	params2_grad[:,1:] += factor * params2[:,1:]

	# Unroll again
	return np.append(params1_grad, params2_grad)

# Returns a gradient vector similar to cost_grad, but computed analytically
# by pertubing each parameter slightly
def cost_grad_numeric(params, input_size, hidden_size, output_size, X, y,
		regularization):
	epsilon = 0.0001

	# Pertube each parameter
	grad = np.zeros(params.size)
	for i in range(0, params.size):
		# Pertube the parameter
		plus = np.copy(params)
		plus[i] += epsilon
		minus = np.copy(params)
		minus[i] -= epsilon

		# Calculate cost and numerically estimate the gradient
		plus_cost = cost_reg(plus, input_size, hidden_size,
			output_size, X, y, regularization)
		minus_cost = cost_reg(minus, input_size, hidden_size,
			output_size, X, y, regularization)
		grad[i] = (plus_cost - minus_cost) / (2.0 * epsilon)
	return grad

# Gradient checks our back propagation algorithm using a small test neural
# network with randomly initialized weights
def grad_check():
	# Create a small neural network, since gradient checking is expensive and we
	# don't want the process to take too long
	input_size = 3
	hidden_size = 5
	output_size = 3
	num_training_examples = 5

	# Randomly initialise the parameters matrices
	params1 = rand_init((hidden_size, input_size + 1))
	params2 = rand_init((output_size, hidden_size + 1))
	params = np.append(params1, params2)

	# Randomly initialise an input vector and output digits vector
	X = rand_init((num_training_examples, input_size))
	X = np.append(np.ones((num_training_examples, 1)), X, axis=1)
	y = np.mod(np.arange(0, num_training_examples), output_size)

	# Compute the analytical and numerical gradients
	analytic = cost_grad_reg(params, input_size, hidden_size, output_size,
		X, y, 1.0)
	numeric = cost_grad_numeric(params, input_size, hidden_size, output_size,
		X, y, 1.0)

	# Compare them as two column vectors
	print(np.append(analytic.reshape(analytic.size, 1),
		numeric.reshape(numeric.size, 1), axis=1))

	# Compare their norm
	difference = np.linalg.norm(analytic - numeric) \
		/ np.linalg.norm(analytic + numeric)
	print("Difference: " + str(difference))

# Uses the set of parameters to predict what digit the given image looks like
def predict(params, layer1_size, layer2_size, layer3_size, image):
	# Unpack the parameter matrices
	params1_shape = (layer2_size, layer1_size + 1)
	params1_size = params1_shape[0] * params1_shape[1]
	params2_shape = (layer3_size, layer2_size + 1)
	params2_size = params2_shape[0] * params2_shape[1]
	params1 = params[0:params1_size].reshape(params1_shape)
	params2 = params[params1_size:].reshape(params2_shape)

	# Forward propagate to get the network's output
	layer1_values = image
	layer2_values = sigmoid(params1.dot(layer1_values))
	layer2_values = np.append(np.array((1.0)), layer2_values)
	layer3_values = sigmoid(params2.dot(layer2_values))
	output = layer3_values

	# The digit we're after is the index of the greatest value in `output`
	return np.argmax(output)

regularization_constant = 1.0

# Load training data
training_data = scipy.io.loadmat("1-data.mat")
X = training_data["X"]
y = training_data["y"]
num_training_examples = X.shape[0]

# To make dealing with MatLab's 1 based indexing easier, the digit 0 is
# represented by 10. Since we're using Python we revert this.
y = np.array([0 if x == 10 else x for x in y])

# Add column of 1s for the bias parameter
X = np.append(np.ones((num_training_examples, 1)), X, axis=1)

# Disable gradient checking, since we know the algorithm works
# grad_check()

# Network architecture
input_size = 400
hidden_size = 25
output_size = 10

# Randomly initialise the parameter matrices
params1 = rand_init((hidden_size, input_size + 1))
params2 = rand_init((output_size, hidden_size + 1))
init_params = np.append(params1, params2)

# Learn the best parameters
best_params = scipy.optimize.fmin_cg(cost_reg, init_params,
	fprime=cost_grad_reg,
	args=(input_size, hidden_size, output_size, X, y, regularization_constant),
	maxiter=50)

# Test the training accuracy of the network
correct = 0
for i in range(0, num_training_examples):
	image = X[i]
	predicted = predict(best_params, input_size, hidden_size, output_size, image)
	if predicted == y[i]:
		correct += 1
print("Accuracy: " + str(float(correct) / num_training_examples * 100.0) + "%")

# Visualise the hidden units (i.e. use the parameters to the hidden layer
# neurons to form 25, 20x20 images)
best_params1 = best_params[0:params1.size].reshape(params1.shape)

# Exclude the column of bias weights
best_params1 = best_params1[:,1:]

# Create a new image of size 5*20 x 5*20
row = 0
column = 0
examples = np.empty((100, 100))
for i in range(0, best_params1.shape[0]):
	inverted_image = best_params1[i,:].reshape(20, 20)
	image = np.transpose(inverted_image)
	examples[row:(row+20),column:(column+20)] = image

	# Update the row and column counters
	if row >= 80:
		row = 0
		column += 20
	else:
		row += 20
plt.imshow(examples, interpolation="nearest")
plt.show()
