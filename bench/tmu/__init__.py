import importlib.util
import logging

logging.disable(logging.INFO)


class SharedContext:
    context = None

    def push(self):
        self.context.push()

    def pop(self):
        self.context.pop()

    def synchronize(self):
        self.context.synchronize()

    def detach(self):
        pass


def make_shared_context(make_context):
    def wrapper(device, flags=0):
        if SharedContext.context is None:
            SharedContext.context = make_context(device, flags)
            return SharedContext.context
        SharedContext.context.push()
        return SharedContext()

    return wrapper


if importlib.util.find_spec("pycuda"):
    import pycuda.driver

    pycuda.driver.Device.make_context = make_shared_context(pycuda.driver.Device.make_context)
