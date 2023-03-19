from graphviz import Digraph

class Value:

    def __init__(self, data, _children=(), _op=''):
        self.data = data # Scalar value
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        x = other.data if isinstance(other, Value) else other
        return Value(self.data + x, (self, other), "+")

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        x = other.data if isinstance(other, Value) else other
        return Value(self.data - x, (self, other), "-")

    def __rsub__(self, other):
        return other - self

    def __mul__(self, other):
        x = other.data if isinstance(other, Value) else other
        return Value(self.data * x, (self, other), "*")

    def __rmul__(self, other):
        return other * self

    def __truediv__(self, other):
        x = other.data if isinstance(other, Value) else other
        return Value(self.data / x, (self, other), "/")

    def __rtruediv__(self, other):
        return other / self

    def __neg__(self):
        return Value(self.data * -1)

def topo_sort(v):
    result = []
    visited = set()

    def recursive_helper(node):
        if node not in visited:
            visited.add(node)
        for child in node._prev:
            seen.add(child)
            recursive_helper(child)
        result.insert(0, node)

    recursive_helper(v)
    return result

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw(root):
    dot = Digraph(format="dot", graph_attr={"rankdir": "LR"}) #Left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        dot.node(name = uid, label = "{data %.4f }" % (n.data, ), shape="record")
        if n._op:
            dot.node(name = uid + n._op, label = n._op)
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot.pipe(encoding="utf-8", format="dot")
