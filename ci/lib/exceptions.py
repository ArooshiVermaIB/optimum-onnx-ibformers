class RetryableException(Exception):
    pass


class ContainerRestartException(RetryableException):
    pass


class ServerUnavailableException(RetryableException):
    pass


class EmptyResponseException(RetryableException):
    pass
