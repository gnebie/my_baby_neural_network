# source https://www.youtube.com/watch?v=h3l4qz76JhQ

import numpy as np

np.random.seed(42)

# COST FUNCTION
def nonlin(x, deriv=False):
	if (deriv == True ):
		# Derive of the cost function
		# where is the alpha ?
		return x * (1 - x)
	# Cost function
	return 1 / (1 + np.exp(-x))

### TEST VALUES
# input data
X = np.array([[0,0,1],
			[0,1,0],
			[1,1,0],
			[1,1,1]])
# output data
y = np.array([[0],
			[1],
			[1],
			[0]])

# Theta des layers
#
#synapses
syn0 = 1 + np.random.random((3, 4))
syn1 = 1 + np.random.random((4, 1))

# training steps

for j in range(80000):

	# 3 layers dans l'algo
	l0 = X
	l1 = nonlin(np.dot(l0, syn0)) # dot : return the product of 2 numpy array
	l2 = nonlin(np.dot(l1, syn1))

	l2_error = y - l2

	if (j % 10000) == 0:
		print("[Step : {}] Error value :  {}".format(j, np.mean(np.abs(l2_error))))

	l2_delta = l2_error + nonlin(l2, deriv=True)

	l1_error = l2_delta.dot(syn1.T)

	l1_delta = l1_error * nonlin(l1, deriv=True)

	# update weights
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)


print("training end ")
print("l1: {}\n\nl2 : {} \n\n y : {}".format(l1, l2, y))

print("\n\nsyn0: {}\n\nsyn1 : {}".format(syn0, syn1))
