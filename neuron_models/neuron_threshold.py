from neuron import neuron_base

class neuron_threshold(neuron_base):
    threshold = 0.5
    def activation(self,x):
        return float(x > self.threshold)


if __name__ == '__main__':
    nt = neuron_threshold(0,10.)

    print(nt([1.]))
