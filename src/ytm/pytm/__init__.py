import logging

PACKAGE_LOGGER_NAME = "pytm"
logger = logging.getLogger(PACKAGE_LOGGER_NAME)
handler = logging.StreamHandler()
# formatter = logging.Formatter("%(message)s")
# handler.setFormatter(formatter)
logger.addHandler(handler)

logger.propagate = False
