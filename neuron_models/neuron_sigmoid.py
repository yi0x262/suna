from neuron import neuron_base
from math import e

class neuron_sigmoid(neuron_base):
    def activation(self,x):
        try:
            return 1/(1+e**-x)
        except:
            return e**x/(e**x + 1)

if __name__ == '__main__':
    ns = neuron_sigmoid(0,1.)

    print(ns([1e-10]))
    print(ns([1e10]))
    print(ns([float('inf')]))
    print(ns([-float('inf')]))
