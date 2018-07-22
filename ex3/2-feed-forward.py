
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
# neurons in the second hidden layer, 10 output neurons (one for each digit) in
# the output layer.

import numpy as np
import scipy.io

# Load training data
training_data = scipy.io.loadmat("2-data.mat")
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
weights_data = scipy.io.loadmat("2-weights.mat")
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

# Predicts which digit (0 - 9) the given image represents
def predict(image):
	# First layer is just the input pixel values
	layer1_values = image

	# Second layer is the 25 hidden neurons
	layer2_values = sigmoid(params1.dot(layer1_values))

	# Add bias parameter to the hidden layer
	layer2_values = np.append(np.array((1.0)), layer2_values)

	# Calculate output layer
	layer3_values = sigmoid(params2.dot(layer2_values))

	# The index of the maximum probability in the ouptut layer is the digit
	# that we're after
	return np.argmax(layer3_values)

# Calculate the accuracy of the neural network over the training data
correct = 0
for row in range(0, num_samples):
	image = features[row]
	predicted = predict(image)
	if predicted == digits[row]:
		correct += 1
print("Accuracy: " + str(float(correct) / float(num_samples) * 100.0) + "%")
