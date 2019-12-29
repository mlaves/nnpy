import numpy as np


class CrossEntropy:
    """
    Inspired by https://towardsdatascience.com/building-an-artificial-neural-network-using-pure-numpy-3fe21acc5815
    """
    def __init__(self):
        self.logits = None
        self.targets = None
        self.grads = None

    def forward(self, logits, targets):
        self.logits = logits
        self.targets = targets

        logits_for_targets = logits[np.arange(len(logits)), self.targets]
        output = -logits_for_targets + np.log(np.sum(np.exp(logits), axis=-1))

        return output

    def backward(self):
        ones_for_targets = np.zeros_like(self.logits)
        ones_for_targets[np.arange(len(self.logits)), self.targets] = 1

        softmax = np.exp(self.logits) / np.exp(self.logits).sum(axis=-1, keepdims=True)

        self.grads = (-ones_for_targets + softmax) / self.logits.shape[0]

        return self.grads


class MSE:
    def __init__(self):
        self.logits = None
        self.targets = None
        self.grads = None

    def forward(self, logits, targets):
        assert logits.shape == targets.shape
        self.logits = logits
        self.targets = targets

        output = np.mean(np.square(self.logits - self.targets))

        return output

    def backward(self):
        self.grads = np.multiply(2/self.logits.shape[-1], self.logits - self.targets)
        return self.grads
