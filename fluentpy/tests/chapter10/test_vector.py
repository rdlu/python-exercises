import math
import pytest

import fluentpy.chapter10.vector as v


class TestVector:
    v1 = v.Vector([3.1, 4.2])
    v2 = v.Vector((3, 4, 5))
    v3 = v.Vector(range(10))

    def test_iter_matching(self):
        x, y = self.v1
        assert (x, y) == (3.1, 4.2), \
            'matches two members'

        p, q, r = self.v2
        assert (p, q, r) == (3.0, 4.0, 5.0), \
            'matches three members'

        first, *rest = self.v3
        assert (first, rest) == (0, list(range(1, 10))), \
            'matches ranges and unpacking'

    def test_repr(self):
        assert repr(self.v1) == 'Vector([3.1, 4.2])', \
            'object is represented for base float case'
        assert repr(self.v2) == 'Vector([3.0, 4.0, 5.0])', \
            'object is represented for integers and tuples'
        assert repr(self.v3) == 'Vector([0.0, 1.0, 2.0, 3.0, 4.0, ...])', \
            'object is represented for ranges'

    def test_clone_eval_eq(self):
        v1_clone = eval('v.' + repr(self.v1))
        assert self.v1 == v1_clone, \
            'repr must be evaluated'

    def test_len(self):
        assert len(self.v1) == 2
        assert len(self.v3) == 10

    def test_frombytes(self):
        v1_clone = v.Vector.frombytes(bytes(self.v1))
        assert v1_clone == self.v1, \
            'object can be serialized in bytes and reversed frombytes'

    def test_get_item(self):
        assert (self.v2[0], self.v2[len(self.v2) - 1], self.v2[-1]) == (3.0, 5.0, 5.0), \
            'we can get items using brackets [] notation'

    def test_slicing(self):
        assert self.v3[1:4] == v.Vector([1.0, 2.0, 3.0]), \
            'we can slice with positive numbers'

        assert self.v3[-1:] == v.Vector([9]), \
            'we can slice negative ranges'

        with pytest.raises(TypeError):
            x = self.v3[1, 2]  # we dont support multidimensional

    def test_str(self):
        assert str(self.v1) == '(3.1, 4.2)'
        assert str(self.v2) == '(3.0, 4.0, 5.0)'

    def test_hashing(self):
        assert hash(self.v2) == hash(v.Vector([3.0, 4.0, 5.0])), \
            'hash must be the same for the same vector components, using floats instead integers'
        assert hash(self.v1) != hash(self.v3), \
            'hash must be different for different components'
