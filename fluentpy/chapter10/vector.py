import functools
import operator
from array import array
import reprlib
import math


class Vector:
    typecode = 'd'

    def __init__(self, components):
        self._components = array(self.typecode, components)

    def __iter__(self):
        return iter(self._components)

    def __repr__(self):
        components = reprlib.repr(self._components)
        components = components[components.find('['):-1]
        return "Vector({})".format(components)

    def __eq__(self, other):
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    def __len__(self):
        return len(self._components)

    def __getitem__(self, index):
        return self._components[index]

    # def __hash__(self):
    #     hashes = (hash(x) for x in self)
    #     return functools.reduce(operator.xor, hashes, 0)

    def __bytes__(self):
        return bytes([ord(self.typecode)]) + bytes(self._components)

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        memv = memoryview(octets[1:]).cast(typecode)
        return cls(memv)
