import numpy as np

import aesara
import aesara.tensor as aet
from aesara.tensor.type import dmatrix, scalars


class Mlp:
    def __init__(self, nfeatures=100, noutputs=10, nhiddens=50, rng=None):
        if rng is None:
            rng = 0
        if isinstance(rng, int):
            rng = np.random.RandomState(rng)
        self.rng = rng
        self.nfeatures = nfeatures
        self.noutputs = noutputs
        self.nhiddens = nhiddens

        x = dmatrix("x")
        wh = aesara.shared(self.rng.normal(0, 1, (nfeatures, nhiddens)), borrow=True)
        bh = aesara.shared(np.zeros(nhiddens), borrow=True)
        h = aesara.tensor.sigmoid(aet.dot(x, wh) + bh)

        wy = aesara.shared(self.rng.normal(0, 1, (nhiddens, noutputs)))
        by = aesara.shared(np.zeros(noutputs), borrow=True)
        y = aesara.tensor.nnet.softmax(aet.dot(h, wy) + by)

        self.inputs = [x]
        self.outputs = [y]


class OfgNested:
    def __init__(self):
        x, y, z = scalars("xyz")
        e = x * y
        op = aesara.compile.builders.OpFromGraph([x, y], [e])
        e2 = op(x, y) + z
        op2 = aesara.compile.builders.OpFromGraph([x, y, z], [e2])
        e3 = op2(x, y, z) + z

        self.inputs = [x, y, z]
        self.outputs = [e3]


class Ofg:
    def __init__(self):
        x, y, z = scalars("xyz")
        e = aesara.tensor.sigmoid((x + y + z) ** 2)
        op = aesara.compile.builders.OpFromGraph([x, y, z], [e])
        e2 = op(x, y, z) + op(z, y, x)

        self.inputs = [x, y, z]
        self.outputs = [e2]


class OfgSimple:
    def __init__(self):
        x, y, z = scalars("xyz")
        e = aesara.tensor.sigmoid((x + y + z) ** 2)
        op = aesara.compile.builders.OpFromGraph([x, y, z], [e])
        e2 = op(x, y, z)

        self.inputs = [x, y, z]
        self.outputs = [e2]
