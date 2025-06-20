import numpy as np
from time import time
from keras.datasets import fashion_mnist

from mltm import MultiClassTM

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
	X_train = X_train.reshape((X_train.shape[0], -1)).astype(np.uint32)

	out = np.zeros((*X_test.shape, ch))
	for j in range(ch):
		t1 = (j + 1) * 255 / (ch + 1)
		out[:, :, :, j] = (X_test >= t1) & 1
	X_test = np.array(out)
	X_test = X_test.reshape((X_test.shape[0], -1)).astype(np.uint32)

	tm = MultiClassTM(
		number_of_clauses=4000,
		T=10000,
		s=10,
		dim=(28, 28, 8),
		patch_dim=(3, 3),
		block_size=256,
	)

	for i in range(5):
		start_training = time()
		tm.fit(X_train, Y_train)
		stop_training = time()

		start_testing = time()
		pred , _ = tm.predict(X_test)
		result_test = 100 * (pred == Y_test).mean()
		stop_testing = time()

		pred_train, _ = tm.predict(X_train)
		result_train = 100 * (pred_train == Y_train).mean()

		print(
			f"Epoch {i + 1} | Train Time: {stop_training - start_training:.2f}s, Test Time: {stop_testing - start_testing:.2f}s | Train Accuracy: {result_train:.4f}, Test Accuracy: {result_test:.4f}"
		)
