class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data # Scalar value
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        x = other.data if isinstance(other, Value) else other
        return Value(self.data + x)

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        x = other.data if isinstance(other, Value) else other
        return Value(self.data - x)

    def __rsub__(self, other):
        return other - self

    def __mul__(self, other):
        x = other.data if isinstance(other, Value) else other
        return Value(self.data * x)

    def __rmul__(self, other):
        return other * self

    def __truediv__(self, other):
        x = other.data if isinstance(other, Value) else other
        return Value(self.data / x)

    def __rtruediv__(self, other):
        return other / self

    def __neg__(self):
        return Value(self.data * -1)
