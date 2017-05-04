from abc import ABCMeta,abstractmethod

class neuron_base(metaclass=ABCMeta):
    def __init__(self,identical_number,adaptation_speed,internal_state=0.):
        """
        id  : int
        adaptation_speed    : float
        internal_state      : float
        """
        self.id = identical_number
        self.adaptation_speed = adaptation_speed
        self.internal_state = internal_state

    def __call__(self,x):
        """v is sum of inputs"""
        self.internal_state += (self.activation(x)-self.internal_state)/self.adaptation_speed
        return self.internal_state

    @staticmethod
    @abstractmethod
    def activation(self,x):
        """x is a scalar"""
        pass

if __name__ == '__main__':
    class neuron_test(neuron_base):
        def activation(self,x):
            return x

    nt = neuron_test(0,1.)

    print(nt([1.,2.]))
