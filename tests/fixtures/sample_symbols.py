"""Sample symbols for extract_symbol / find_symbol unit tests."""


def plain_helper(value: int) -> int:
    return value + 1


def _noop_decorator(fn):
    return fn


@_noop_decorator
@_noop_decorator
def decorated_helper(value: int) -> int:
    """Decorated so start_line includes decorator lines."""
    return value * 2


class SampleWidget:
    def method_one(self) -> str:
        return "one"
