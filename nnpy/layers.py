import numpy as np
from abc import abstractmethod


class Layer:
    @abstractmethod
    def __init__(self):
        self.training = True
        self.name = 'Layer'
        pass

    def parameters(self):
        return
        yield

    def gradients(self):
        return
        yield

    @abstractmethod
    def forward(self, x):
        pass

    def backward(self, previous_grad):
        return previous_grad


class Linear(Layer):
    def __init__(self, input_units, output_units):
        super().__init__()
        self.name = 'Linear'
        self.input_units = input_units
        self.output_units = output_units
        self.inputs = None
        self.weight = (np.random.randn(output_units, input_units) * 0.01).astype(np.float32)
        self.bias = np.zeros(output_units, dtype=np.float32)
        self.weight_grad = np.zeros((output_units, input_units), dtype=np.float32)
        self.bias_grad = np.zeros(output_units, dtype=np.float32)

    def parameters(self):
        yield self.weight
        yield self.bias

    def gradients(self):
        yield self.weight_grad
        yield self.bias_grad

    def forward(self, x):
        assert len(x.shape) == 2 or len(x.shape) == 3

        self.inputs = x.copy()
        input_shape = self.inputs.shape

        if len(input_shape) == 3:
            b, c, _ = input_shape
            output_shape = (b, c, self.output_units)
        else:
            b, _ = input_shape
            output_shape = (b, self.output_units)

        return np.dot(self.inputs.reshape(-1, self.input_units),
                      self.weight.transpose()).reshape(output_shape) + self.bias

    def backward(self, previous_grad):
        assert len(previous_grad.shape) == 2 or len(previous_grad.shape) == 3

        if len(previous_grad.shape) == 3:
            sum_axis = (1, 0)
        else:
            sum_axis = (0,)

        b = previous_grad.shape[0]
        grad_input = np.dot(previous_grad.reshape(-1, self.output_units), self.weight).reshape(self.inputs.shape)

        self.weight_grad = np.dot(np.transpose(previous_grad).reshape(-1, b),
                                  self.inputs.reshape(b, -1)).reshape(self.weight_grad.shape)
        self.bias_grad = np.sum(previous_grad, axis=sum_axis)

        return grad_input


class Flatten(Layer):
    def __init__(self, keep_channels=False):
        super().__init__()
        self.name = 'Flatten'
        self.keep_channels = keep_channels
        self.forward_shape = None

    def forward(self, x):
        self.forward_shape = x.shape
        if self.keep_channels:
            return x.reshape(x.shape[0], x.shape[1], -1)
        else:
            return x.reshape(x.shape[0], -1)

    def backward(self, previous_grad):
        return previous_grad.reshape(self.forward_shape)


class Dropout(Layer):
    """
    Elementwise dropout (inverted dropout).

    Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., Salakhutdinov, R.
    Dropout: A Simple Way to Prevent Neural Networks from Overfitting.
    Journal of Machine Learning Research, 15(1), 1929–1958. (2014)
    """
    def __init__(self, p=0.5):
        """

        :param p: Probability of keeping a unit active (higher = less dropout).
        """
        super().__init__()
        self.name = 'Dropout'
        self.p = p

    def parameters(self):
        return
        yield

    def gradients(self):
        return
        yield

    def forward(self, x):
        if self.training:
            dropout = np.random.binomial(1, self.p, x.shape)
            return np.divide(np.multiply(x, dropout), self.p)  # We divide by p to compensate for deactivated units
        else:
            return x


class BatchNorm(Layer):
    """
    Implements Batch Normalization over C dimension.

    https://arxiv.org/abs/1502.03167
    """
    def __init__(self, channels, momentum=0.9, eps=1e-6):
        super().__init__()
        self.name = 'BatchNorm'
        self.momentum = momentum

        self.gamma = np.ones((channels,), dtype=np.float32)
        self.beta = np.zeros((channels,), dtype=np.float32)
        self.grad_gamma = np.zeros((channels,), dtype=np.float32)
        self.grad_beta = np.zeros((channels,), dtype=np.float32)

        self.eps = np.array([eps], dtype=np.float32)

        self.x = np.zeros([], dtype=np.float32)
        self.mean = np.array([], dtype=np.float32)
        self.var = np.array([], dtype=np.float32)
        self.xhat = np.array([], dtype=np.float32)
        self.running_mean = np.array([0], dtype=np.float32)
        self.running_var = np.array([0], dtype=np.float32)

    def parameters(self):
        yield self.gamma
        yield self.beta

    def gradients(self):
        yield self.grad_gamma
        yield self.grad_beta

    def forward(self, x):
        assert len(x.shape) == 2 or len(x.shape) == 3
        self.x = x.copy()
        in_l = self.x.shape[-1]

        if self.training:
            if len(self.x.shape) == 3:
                mean_var_axis = (2, 0)
            else:
                mean_var_axis = (0,)

            self.mean = np.mean(self.x, axis=mean_var_axis, keepdims=True)
            self.var = np.var(self.x, axis=mean_var_axis, keepdims=True)

            # normalize
            self.xhat = (self.x - self.mean) / np.sqrt(self.var + self.eps)

            # scale and shift
            if len(self.x.shape) == 3:
                y = self.xhat * self.gamma[:, np.newaxis].repeat(in_l, axis=-1) + self.beta[:, np.newaxis].repeat(in_l, axis=-1)
            else:
                y = self.xhat * self.gamma + self.beta

            self.running_mean = self.mean * (1-self.momentum) + self.running_mean * self.momentum
            self.running_var = self.var * (1-self.momentum) + self.running_var * self.momentum
        else:  # test time
            # normalize
            xhat = (self.x - self.running_mean) / np.sqrt(self.running_var + self.eps)

            # scale and shift
            if len(self.x.shape) == 3:
                y = xhat * self.gamma[:, np.newaxis].repeat(in_l, axis=-1) + self.beta[:, np.newaxis].repeat(in_l, axis=-1)
            else:
                y = xhat * self.gamma + self.beta

        return y

    def backward(self, previous_grad):
        # calculate derivatives (it's in the paper!)
        m = self.x.shape[0]
        in_l = self.x.shape[-1]

        if len(self.x.shape) == 3:
            d_dxhat = previous_grad * self.gamma[:, np.newaxis].repeat(in_l, axis=-1)
        else:
            d_dxhat = previous_grad * self.gamma

        d_dvar = np.sum(self.x - self.mean, axis=0) * d_dxhat * (-0.5) * (self.var + self.eps) ** (-3 / 2)
        d_dmean = np.sum(d_dxhat * -1 / np.sqrt(self.var + self.eps), axis=0) + d_dvar * np.sum(-2 * (self.x - self.mean), axis=0) / m

        d_dx = d_dxhat * 1 / np.sqrt(self.var + self.eps) + d_dvar * 2 * (self.x - self.mean) / m + d_dmean / m

        if len(previous_grad.shape) == 3:
            sum_axis = (0, 2)
        else:
            sum_axis = 0

        self.grad_gamma = np.sum(np.multiply(previous_grad, self.xhat), axis=sum_axis).transpose()
        self.grad_beta = np.sum(previous_grad, axis=sum_axis).transpose()

        return d_dx
