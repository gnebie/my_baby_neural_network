#source : https://www.youtube.com/watch?v=cdLUzrjnlr4

import copy, numpy as np

# nonlitearaty sigmoid

def sigmoid(x):
	sig = 1 / (1 + np.exp(-x))
	return sig

# sigmoid derivative

def sigmoid_derivative(x):
	return x * (1 - x)



def main():
	np.random.seed(42)
	int2binary = {}
	binary_dim = 8

	largers_num = pow(2, binary_dim)
	binary = np.unpackbits(
		np.array(
				[range(largers_num)], dtype=np.uint8).T, axis=1
		)
	for i in range(largers_num):
		int2binary[i] = binary[i]

	alpha = 0.3
	input_dim = 2
	hidden_dim = 10
	output_dim = 1

	# initialize neural network weights
	synapse_0 = 2*np.random.random((input_dim,hidden_dim)) - 1
	synapse_1 = 2*np.random.random((hidden_dim,output_dim)) - 1
	synapse_h = 2*np.random.random((hidden_dim,hidden_dim)) - 1

	synapse_0_update = np.zeros_like(synapse_0)
	synapse_1_update = np.zeros_like(synapse_1)
	synapse_h_update = np.zeros_like(synapse_h)


	for j in range(100000):
		# let's do an addition
		a_int = np.random.randint(largers_num / 2)
		a = int2binary[a_int]

		b_int = np.random.randint(largers_num / 2)
		b = int2binary[b_int]

		c_int = a_int + b_int
		c = int2binary[c_int]

		d = np.zeros_like(c)

		overallError = 0
		layer_2_deltas = list()
		layer_1_values = list()
		layer_1_values.append(np.zeros(hidden_dim))

		for position in range(binary_dim):
			# input and output
			X = np.array([[a[binary_dim - position - 1], b[binary_dim - position - 1]]])
			y = np.array([[c[binary_dim - position - 1]]]).T

			# hidden layer
			layer_1 = sigmoid(np.dot(X, synapse_0) + np.dot(layer_1_values[-1], synapse_h))
			# output layer
			layer_2 = sigmoid(np.dot(layer_1, synapse_1))

			# error :
			layer_2_error = y - layer_2
			layer_2_deltas.append((layer_2_error) * sigmoid_derivative(layer_2))
			overallError += np.abs(layer_2_error[0])

			d[binary_dim - position - 1] = np.round(layer_2[0][0])
			layer_1_values.append(copy.deepcopy(layer_1))

		future_layer_1_delta = np.zeros(hidden_dim)

		for position in range(binary_dim):
			X = np.array([[a[position], b[position]]])
			layer_1 = layer_1_values[-position - 1]
			prev_layer_1 = layer_1_values[-position - 2]

			# error at output
			layer_2_delta = layer_2_deltas[-position - 1]

			# error at hidden layer
			layer_1_delta = (future_layer_1_delta.dot(synapse_h.T) + layer_2_delta.dot(synapse_1.T)) * sigmoid_derivative(layer_1)

			synapse_0_update += X.T.dot(layer_1_delta)
			synapse_1_update += np.atleast_2d(layer_1).T.dot(layer_2_delta)
			synapse_h_update += np.atleast_2d(prev_layer_1).T.dot(layer_1_delta)

			future_layer_1_delta = layer_1_delta

		synapse_0 += (synapse_0_update * alpha)
		synapse_1 += (synapse_1_update * alpha)
		synapse_h += (synapse_h_update * alpha)

		synapse_0_update *= 0
		synapse_1_update *= 0
		synapse_h_update *= 0

		if (j % 1000) == 0:
			# print(j)
			print("Error: {}".format(overallError))
			# print("pred: {}".format(d))
			# print("true: {}".format(c))
			out = 0
			# for index, x in enumerate(reversed(d)):
			# 	out += x * pow(2, index)
			# print("{} + {} = {}".format(a_int, b_int, out))

if __name__ == '__main__':
	main()
