import logging

# Added: Create logger and assign handler
def get_logger(name, verbose=True):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    formatter = logging.Formatter(
        fmt="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
