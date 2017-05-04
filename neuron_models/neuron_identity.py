from .neuron import neuron_base

class neuron_identity(neuron_base):
    def activation(self,x):
        return x

if __name__ == '__main__':
    ni = neuron_identity(0,10.)
    print(ni([1]))
