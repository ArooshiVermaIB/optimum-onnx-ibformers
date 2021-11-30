class RetryableException(Exception):
    pass


class ContainerRestartException(RetryableException):
    pass
