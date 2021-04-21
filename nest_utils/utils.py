from typing import Any, Iterator

from argparse import Namespace as BaseNamespace


class Context(BaseNamespace):
    """Helper class for storing module context.
    """

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, val: str) -> Any:
        return setattr(self, key, val)

    def __iter__(self) -> Iterator:
        return iter(self.__dict__.items())

    def items(self) -> Iterator:
        return self.__dict__.items()

    def keys(self) -> Iterator:
        return self.__dict__.keys()

    def values(self) -> Iterator:
        return self.__dict__.values()

    def clear(self):
        self.__dict__.clear()
