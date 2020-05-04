import functools
import itertools
import math
import numbers
import operator
import reprlib
from array import array


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
        cls = type(self)
        if isinstance(index, slice):
            return cls(self._components[index])
        elif isinstance(index, numbers.Integral):
            return self._components[index]
        else:
            msg = '{.__name__} indices must be integers'
            raise TypeError(msg.format(cls))

    def __str__(self):
        return str(tuple(self))

    def __format__(self, format_spec=''):
        if format_spec.endswith('h'):
            # n-sphere coords
            format_spec = format_spec[:-1]
            outer_format = '<{}>'
            coords = itertools.chain([abs(self)], self.angles())
        else:
            outer_format = '({})'
            coords = self

        components = (format(component, format_spec) for component in coords)
        return outer_format.format(', '.join(components))

    def angle(self, n):
        r = math.sqrt(sum(x * x for x in self[n:]))
        a = math.atan2(r, self[n - 1])
        if (n == len(self) - 1) and (self[-1] < 0):
            return math.pi * 2 - a
        else:
            return a

    def angles(self):
        return (self.angle(n) for n in range(1, len(self)))

    def __hash__(self):
        hashes = (hash(x) for x in self)
        return functools.reduce(operator.xor, hashes, 0)

    def __abs__(self):
        return math.sqrt(sum(x ** 2 for x in self))

    def __bytes__(self):
        return bytes([ord(self.typecode)]) + bytes(self._components)

    @classmethod
    def frombytes(cls, octets):
        typecode = chr(octets[0])
        mem_view = memoryview(octets[1:]).cast(typecode)
        return cls(mem_view)
