import math
import pytest

import fluentpy.chapter10.vector as v


class TestVector:
    v1 = v.Vector([3.1, 4.2])
    v2 = v.Vector((3, 4, 5))
    v3 = v.Vector(range(10))

    def test_repr(self):
        assert repr(self.v1) == 'Vector([3.1, 4.2])', \
            'object is represented for base float case'
        assert repr(self.v2) == 'Vector([3.0, 4.0, 5.0])', \
            'object is represented for integers and tuples'
        assert repr(self.v3) == 'Vector([0.0, 1.0, 2.0, 3.0, 4.0, ...])', \
            'object is represented for ranges'

    def test_clone_eval(self):
        v1_clone = eval('v.' + repr(self.v1))
        assert self.v1 == v1_clone, \
            'repr must be evaluated'

    def test_len(self):
        assert len(self.v1) == 2
        assert len(self.v3) == 10

    def test_frombytes(self):
        v1_clone = v.Vector.frombytes(bytes(self.v1))
        assert v1_clone == self.v1

    def test_get_item(self):
        assert (self.v2[0], self.v2[len(self.v2) - 1], self.v2[-1]) == (3.0, 5.0, 5.0)
