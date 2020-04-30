import fluentpy.chapter9.vector2d as v2d
import math
import pytest


class TestVector2d:
    v1 = v2d.Vector2d(3, 4)

    def test_init_int(self):
        x, y = self.v1
        assert (x, y) == (3.0, 4.0), 'Must return a tuple'

    def test_clone_eval(self):
        v1_clone = eval('v2d.' + repr(self.v1))
        assert self.v1 == v1_clone, \
            'repr must be evaluated'

    def test_str(self):
        assert str(self.v1) == '(3.0, 4.0)', \
            'str conversion need to print a tuple'

    def test_bytes(self):
        assert bytes(self.v1) == b'd\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\x10@', \
            'byte representation must match'

    def test_abs(self):
        assert abs(self.v1) == 5.0, \
            'abs must be working'

    def test_bool(self):
        assert (bool(self.v1), bool(v2d.Vector2d(0, 0))) == (True, False), \
            'bool conversion must evaluate False on null vectors'

    def test_from_bytes(self):
        v1_clone = v2d.Vector2d.frombytes(bytes(self.v1))
        assert self.v1 == v1_clone

    def test_format(self):
        assert format(self.v1) == '(3.0, 4.0)', \
            'format without params'
        assert format(self.v1, '.2f') == '(3.00, 4.00)', \
            'format 2 decimal places'
        assert format(self.v1, '.3e') == '(3.000e+00, 4.000e+00)', \
            'format cientific exp mode'

    def test_angle(self):
        assert v2d.Vector2d(0, 0).angle() == 0.0, \
            'angle on null vector is zero'
        assert v2d.Vector2d(1, 0).angle() == 0.0, \
            'angle with y component zero is zero'
        epsilon = 10 ** -8
        assert abs(v2d.Vector2d(0, 1).angle() - math.pi / 2) < epsilon
        assert abs(v2d.Vector2d(1, 1).angle() - math.pi / 4) < epsilon

    def test_format_polar(self):
        v_1_1 = v2d.Vector2d(1, 1)
        assert format(v_1_1, 'p') == '<1.4142135623730951, 0.7853981633974483>', \
            'vector in polar format must match'
        assert format(v_1_1, '.3ep') == '<1.414e+00, 7.854e-01>', \
            'vector in polar format, with 3 decimals and exponential rest must match'
        assert format(v_1_1, '0.5fp') == '<1.41421, 0.78540>', \
            'vector in polar format, with 5 decimals, truncated rest must match'

    def test_x_y_handling(self):
        # x and y must be read only
        with pytest.raises(AttributeError):  # as e_info:
            self.v1.x = 1
            self.v1.y = 2

    def test_hashing(self):
        v2 = v2d.Vector2d(3.1, 4.2)
        assert hash(self.v1) == 7, \
            'hash of vector (3, 4) is 7'
        assert hash(v2) == 384307168202284039, \
            'hash of a non integer components must work'
        assert hash(self.v1) != hash(v2), \
            'hash of (3, 4) must not match to (3.n, 4.n)'
        assert hash(self.v1) == hash(v2d.Vector2d(3, 4)), \
            'two different instance of vectors with same components must match'

    def test_set(self):
        assert len({self.v1, v2d.Vector2d(0, 0)}) == 2, \
            'converting to a set must work, with two different vectors'
        assert len({self.v1, self.v1}) == 1, \
            'converting to set must work, with two equivalent vectors'

