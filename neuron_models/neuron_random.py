from neuron import neuron_base
from random import uniform

class neuron_random(neuron_base):
    def activation(self,x):
        return uniform(0,1)


if __name__ == '__main__':
    nr = neuron_random(0,1.)
    print(nr([1]))
