import numpy as np
from .layers import Layer


class ReLU(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'ReLU'
        self.inputs = None

    def forward(self, x):
        self.inputs = x.copy()
        return np.maximum(0.0, self.inputs)

    def backward(self, previous_grad):
        grad_input = (self.inputs > 0.0).astype(np.float32)
        return previous_grad*grad_input


class Sigmoid(Layer):
    def __init__(self):
        super().__init__()
        self.name = 'Sigmoid'
        self.outputs = None

    def forward(self, x):
        exp_neg_x = np.exp(-x)
        self.outputs = np.divide(1, 1 + exp_neg_x)
        return self.outputs

    def backward(self, previous_grad):
        grad_input = self.outputs * (1 - self.outputs)
        return previous_grad*grad_input
