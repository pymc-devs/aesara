import numpy as np

from aesara.compile.function.pfunc import pfunc
from aesara.compile.sharedvalue import shared
from aesara.gradient import grad
from aesara.tensor.math import dot, sigmoid
from aesara.tensor.math import sum as aet_sum
from aesara.tensor.type import dvector


class NNet:
    def __init__(
        self,
        input=None,
        target=None,
        n_input=1,
        n_hidden=1,
        n_output=1,
        lr=1e-3,
        **kw,
    ):
        super().__init__(**kw)

        if input is None:
            input = dvector("input")
        if target is None:
            target = dvector("target")

        self.input = input
        self.target = target
        self.lr = shared(lr, "learning_rate")
        self.w1 = shared(np.zeros((n_hidden, n_input)), "w1")
        self.w2 = shared(np.zeros((n_output, n_hidden)), "w2")
        # print self.lr.type

        self.hidden = sigmoid(dot(self.w1, self.input))
        self.output = dot(self.w2, self.hidden)
        self.cost = aet_sum((self.output - self.target) ** 2)

        self.sgd_updates = {
            self.w1: self.w1 - self.lr * grad(self.cost, self.w1),
            self.w2: self.w2 - self.lr * grad(self.cost, self.w2),
        }

        self.sgd_step = pfunc(
            params=[self.input, self.target],
            outputs=[self.output, self.cost],
            updates=self.sgd_updates,
        )

        self.compute_output = pfunc([self.input], self.output)

        self.output_from_hidden = pfunc([self.hidden], self.output)


def test_nnet():
    rng = np.random.RandomState(1827)
    data = rng.rand(10, 4)
    nnet = NNet(n_input=3, n_hidden=10)
    for epoch in range(3):
        mean_cost = 0
        for x in data:
            input = x[0:3]
            target = x[3:]
            output, cost = nnet.sgd_step(input, target)
            mean_cost += cost
        mean_cost /= float(len(data))
        # print 'Mean cost at epoch %s: %s' % (epoch, mean_cost)
    assert abs(mean_cost - 0.20588975452) < 1e-6
    # Just call functions to make sure they do not crash.
    nnet.compute_output(input)
    nnet.output_from_hidden(np.ones(10))
