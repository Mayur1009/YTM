from time import time

class Timer:
    start_time: float
    end_time: float

    def __init__(self, logger=None, text=None):
        self.text = text if text else ""
        self.logger = logger

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time()

    def elapsed(self):
        if self.end_time is None:
            raise RuntimeError("elapsed() must be called after context is ended.")
        return self.end_time - self.start_time
