# FIXME: Broken because of patch_weights and other things.
class AutoEncoderTsetlinMachine(CommonTsetlinMachine):
	def __init__(
		self,
		number_of_clauses,
		T,
		s,
		active_output,
		q=1.0,
		max_included_literals=None,
		accumulation=1,
		boost_true_positive_feedback=1,
		number_of_state_bits=8,
		append_negated=True,
		r: float = 1.0,
		sr: float | None = None,
		max_weight: int | None = None,
		grid=(16 * 13, 1, 1),
		block=(128, 1, 1),
	):
		super().__init__(
			number_of_clauses,
			T,
			s,
			q=q,
			max_included_literals=max_included_literals,
			boost_true_positive_feedback=boost_true_positive_feedback,
			number_of_state_bits=number_of_state_bits,
			append_negated=append_negated,
			r=r,
			sr=sr,
			max_weight=max_weight,
			grid=grid,
			block=block,
		)
		self.negative_clauses = 1

		self.active_output = np.array(active_output).astype(np.uint32)
		self.accumulation = accumulation

	# FIXME:
	def _init_fit(self, X_csr, encoded_Y, incremental):  # pyright: ignore[reportIncompatibleMethodOverride]
		if not self.initialized:
			self._init(X_csr)  # pyright: ignore[reportAttributeAccessIssue]
			self.prepare(
				g.state,
				self.ta_state_gpu,
				self.clause_weights_gpu,
				self.class_sum_gpu,
				grid=self.grid,
				block=self.block,
			)
			ctx.synchronize()

		elif not incremental:
			self.prepare(
				g.state,
				self.ta_state_gpu,
				self.clause_weights_gpu,
				self.class_sum_gpu,
				grid=self.grid,
				block=self.block,
			)
			ctx.synchronize()

		if not np.array_equal(self.X_train, np.concatenate((X_csr.indptr, X_csr.indices))):
			self.train_X = np.concatenate((X_csr.indptr, X_csr.indices))

			X_csc = X_csr.tocsc()

			self.X_train_csr_indptr_gpu = mem_alloc(X_csr.indptr.nbytes)
			memcpy_htod(self.X_train_csr_indptr_gpu, X_csr.indptr)

			self.X_train_csr_indices_gpu = mem_alloc(X_csr.indices.nbytes)
			memcpy_htod(self.X_train_csr_indices_gpu, X_csr.indices)

			self.X_train_csc_indptr_gpu = mem_alloc(X_csc.indptr.nbytes)
			memcpy_htod(self.X_train_csc_indptr_gpu, X_csc.indptr)

			self.X_train_csc_indices_gpu = mem_alloc(X_csc.indices.nbytes)
			memcpy_htod(self.X_train_csc_indices_gpu, X_csc.indices)

			self.encoded_Y_gpu = mem_alloc(encoded_Y.nbytes)
			memcpy_htod(self.encoded_Y_gpu, encoded_Y)

			self.active_output_gpu = mem_alloc(self.active_output.nbytes)
			memcpy_htod(self.active_output_gpu, self.active_output)

	def _fit(self, X_csr, encoded_Y, number_of_examples, epochs, incremental=False):  # pyright: ignore[reportIncompatibleMethodOverride]
		self._init_fit(X_csr, encoded_Y, incremental=incremental)

		for epoch in range(epochs):
			for e in range(number_of_examples):
				class_sum = np.zeros(self.number_of_outputs).astype(np.int32)
				memcpy_htod(self.class_sum_gpu, class_sum)

				target = np.random.choice(self.number_of_outputs)
				self.produce_autoencoder_examples.prepared_call(
					self.grid,
					self.block,
					g.state,
					self.active_output_gpu,
					self.active_output.shape[0],
					self.X_train_csr_indptr_gpu,
					self.X_train_csr_indices_gpu,
					X_csr.shape[0],
					self.X_train_csc_indptr_gpu,
					self.X_train_csc_indices_gpu,
					X_csr.shape[1],
					self.encoded_X_gpu,
					self.encoded_Y_gpu,
					target,
					int(self.accumulation),
					int(self.T),
					int(self.append_negated),
				)
				ctx.synchronize()

				self.evaluate_update.prepared_call(
					self.grid,
					self.block,
					self.ta_state_gpu,
					self.clause_weights_gpu,
					self.class_sum_gpu,
					self.encoded_X_gpu,
				)
				ctx.synchronize()

				self.update.prepared_call(
					self.grid,
					self.block,
					g.state,
					self.ta_state_gpu,
					self.clause_weights_gpu,
					self.class_sum_gpu,
					self.encoded_X_gpu,
					self.encoded_Y_gpu,
					np.int32(0),
				)
				ctx.synchronize()

		self.ta_state = np.array([])
		self.clause_weights = np.array([])

		return

	def fit(self, X, number_of_examples=2000, epochs=100, incremental=False):
		X_csr = csr_matrix(X)

		self.number_of_outputs = self.active_output.shape[0]

		self.dim = (X_csr.shape[1], 1, 1)
		self.patch_dim = (X_csr.shape[1], 1)

		self.max_y = None
		self.min_y = None

		encoded_Y = np.zeros(self.number_of_outputs, dtype=np.int32)

		self._fit(X_csr, encoded_Y, number_of_examples, epochs, incremental=incremental)

		return

	def score(self, X):
		X = csr_matrix(X)
		return self._score(X)

	def predict(self, X, return_class_sums=False):
		class_sums = self.score(X)
		preds = np.argmax(class_sums, axis=1)
		if return_class_sums:
			return preds, class_sums
		else:
			return preds
