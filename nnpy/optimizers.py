import numpy as np


class SGD:
    def __init__(self, net, learn_rate=0.01, momentum=0.01):
        self.net = net
        self.lr = learn_rate
        self.momentum = momentum
        self.velocities = []

        for layer in net:
            velocity = []
            for param in layer.parameters():
                velocity.append(np.zeros_like(param))
            self.velocities.append(velocity)

    def step(self):
        for layer, velocities in zip(self.net, self.velocities):
            for param, grad, velocity in zip(layer.parameters(), layer.gradients(), velocities):
                # print(layer.name, param.shape, grad.shape, velocity.shape)
                param -= self.lr * grad + self.momentum * velocity
                velocity *= self.momentum
                velocity -= self.lr * grad
