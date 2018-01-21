# coding: utf-8
import numpy as np
import matplotlib.pylab as plt

#ReLU関数        
def relu(x):
    return np.maximum(0, x)

class Relu:

    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
#        self.mask = relu(x)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
