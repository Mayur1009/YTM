import numpy as np
from time import time
from keras.datasets import fashion_mnist

from pytm.tm import MultiClassConvolutionalTsetlinMachine2D

if __name__ == "__main__":
	(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
	X_train = np.copy(X_train)
	X_test = np.copy(X_test)

	ch = 8

	out = np.zeros((*X_train.shape, ch))
	for j in range(ch):
		t1 = (j + 1) * 255 / (ch + 1)
		out[:, :, :, j] = (X_train >= t1) & 1
	X_train = np.array(out)
	X_train = X_train.reshape((X_train.shape[0], -1))

	out = np.zeros((*X_test.shape, ch))
	for j in range(ch):
		t1 = (j + 1) * 255 / (ch + 1)
		out[:, :, :, j] = (X_test >= t1) & 1
	X_test = np.array(out)
	X_test = X_test.reshape((X_test.shape[0], -1))

	tm = MultiClassConvolutionalTsetlinMachine2D(
		number_of_clauses=4000,
		T=10000,
		s=10,
		dim=(28, 28, 8),
		patch_dim=(3, 3),
		grid=(256, 1, 1),
		block=(128, 1, 1),
	)

	for i in range(5):
		start_training = time()
		tm.fit(X_train, Y_train, epochs=1, incremental=True)
		stop_training = time()

		start_testing = time()
		result_test = 100 * (tm.predict(X_test) == Y_test).mean()
		stop_testing = time()

		result_train = 100 * (tm.predict(X_train) == Y_train).mean()

		print(
			f"Epoch {i + 1} | Train Time: {stop_training - start_training:.2f}s, Test Time: {stop_testing - start_testing:.2f}s | Train Accuracy: {result_train:.4f}, Test Accuracy: {result_test:.4f}"
		)
