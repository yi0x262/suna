import numpy as np

from neuron_models import *
neuron_names = [n for n in dir() if 'neuron_' in n]


class input_identity(neuron_identity):
    def input(self, x):
        self.internal_state = x

    def __call__(self, x):
        return self.internal_state


class output_identity(neuron_identity):
    pass


class connections_list(list):
    def __init__(self, num, *connections):
        """connections : (from,to,weight),(,,),..."""
        super().__init__(connections)
        self.num = num

        self.make_adjacent()

    def make_adjacent(self):
        self.adjacent = np.zeros((self.num, self.num))
        for fr, to, weight in self:
            self.adjacent[fr, to] = float(weight)


class individual(list):
    """spectrum, next_state"""
    def __init__(self, inout, connections_list, *neurons):
        super().__init__([input_identity(i, 1.0) for i in range(inout[0])] + [output_identity(i, 1.0) for i in range(inout[0], inout[1])] + list(neurons))
        if len(self) != connections_list.num:
            raise RuntimeError('adjacent matrix not matching neurons')
        self.inout = inout
        self.connections_list = connections_list
        self.spectrum = self.get_spectrum(*neurons)

    def update(self, x):
        i, o = self.inout
        """x:input"""
        assert len(x) == i, 'invalid input'

#       input
        for j in range(i):
            self[j].input(x[j])
#       update
        states = [n.internal_state for n in self]
        tmp = self.connections_list.adjacent.dot(states)
#       output
        return [n(j) for n, j in zip(self, tmp)][i:i+o]

    @staticmethod
    def get_spectrum(*neurons):
        types = [0]*len(neuron_names)
        for n in neurons:
            types[neuron_names.index(n.__class__.__name__)] += 1
        slower = sum([n.adaptation_speed > 1 for n in neurons])
        return types+[slower]


if __name__ == '__main__':
    cl = connections_list(4, (1,2,2), (2,2,1), (3,2,1), (0,1,2), (1,0,2))
    print(cl.adjacent)

    i = individual((1,1), cl, neuron_random(2.,7.), neuron_random(3.,1.))
    print(i.spectrum)
    for _ in range(10):
        print(i.update([1.]))
