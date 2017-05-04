from .neuron import neuron_base

class neuron_control(neuron_base):
    def __init__(self,*args):
        super().__init__(*args)
        self.internal_state = 1.
    def activation(self,x):
        pass
    def __call__(self,x):
        return self.internal_state

if __name__ == '__main__':
    nc = neuron_control(0,1.)
    print(nc(1))
