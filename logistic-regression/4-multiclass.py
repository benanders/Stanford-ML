
# Multiclass logistic regression classifier
#
# The task is to take a subset of images from the MNIST dataset and classify
# them as one of the digits from 0 to 9. The images are 20x20 pixel grayscale
# images, with intensity values represented as floats between 0 and 1. All
# images have been normalized to a constant background and text color.

import numpy as np
import numpy.random
import matplotlib.pyplot as plt
import scipy.io

# `images` is a 5000x400 matrix; each row is a flattened 20x20 grayscale image
# representing a single training example.
#
# `digits` is a 5000 element vector of digits 0 to 9 representing the correct
# answer for each training example.
data = scipy.io.loadmat("data-3.mat")
images = data["X"]
digits = data["y"]

# To make dealing with MatLab's 1 based indexing easier, the digit 0 is
# represented by 10. Since we're using Python we revert this.
digits = np.array([0 if x == 10 else x for x in digits])

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
plt.show()
