from time import time


class Timer:
    """Context manager for measuring elapsed wall-clock time.

    Examples
    --------
    >>> timer = Timer()
    >>> with timer:
    ...     do_work()
    >>> print(timer.elapsed)
    """

    start_time: float
    end_time: float

    def __init__(self, logger=None, text=None):
        self.text = text if text else ""
        self.logger = logger

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time()

    @property
    def elapsed(self) -> float:
        """Elapsed time in seconds. Must be accessed after the context exits.

        Returns
        -------
        float
            Wall-clock seconds between ``__enter__`` and ``__exit__``.

        Raises
        ------
        RuntimeError
            If accessed before the context has exited.
        """
        if self.end_time is None:
            raise RuntimeError("elapsed must be accessed after context is ended.")
        return self.end_time - self.start_time
