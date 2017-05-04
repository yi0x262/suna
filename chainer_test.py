#/usr/bin/env python3

import numpy as np
from chainer import FunctionSet,Variable,optimizers

x = Variable(np.array([0],dtype=np.float32))

y = x**2 - 2*x + 1
print(y.data)

y.backward()
print(x.grad)
