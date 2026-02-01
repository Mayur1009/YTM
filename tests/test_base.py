import numpy as np
from ytm.dev.tm import TM

if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=np.uint32)

    Y = np.array([0, 1, 1, 0], dtype=np.uint32)

    tm = TM(4, 4, 10, (2, 1, 1), 2)

    encoded_X = tm.encode(X)

    print(f'{encoded_X.shape=}')
