class DimensionException(Exception):
    pass


class RangeException(Exception):
    pass


class TypeException(Exception):
    pass


class FunctionDefinitionAssertionError(Exception):
    def __init__(
        self, message="Function definition parameters failed constraint requirements."
    ):
        self.message = message
        super().__init__(self.message)
