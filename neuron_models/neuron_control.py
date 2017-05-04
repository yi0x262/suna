from .neuron import neuron_base

class neuron_control(neuron_base):
    def activation(self,x):
        pass
    def __call__(self,x):
        return 1

if __name__ == '__main__':
    nc = neuron_control(0,1.)
    print(nc(1))
