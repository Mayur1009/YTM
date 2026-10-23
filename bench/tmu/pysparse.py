from scipy.sparse import csr_matrix


class PySparseTM:
    def __init__(self, tm):
        self.tm = tm

    def fit(self, X, Y):
        self.tm.fit(X, Y, epochs=1, incremental=True)

    def predict(self, X):
        return self.tm.predict(X)


def to_csr(X_train, Y_train, X_test, Y_test):
    return csr_matrix(X_train.reshape(len(X_train), -1)), Y_train, csr_matrix(X_test.reshape(len(X_test), -1)), Y_test
