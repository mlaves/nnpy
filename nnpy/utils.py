import numpy as np


def forward(net, x):
    for layer in net:
        x = layer.forward(x)
        # print('f', layer.name, x.shape)
    return x


def backward(net, loss):
    grad = loss.backward()
    for layer in reversed(net):
        grad = layer.backward(grad)
        # print('b', layer.name, grad.shape)


def dataloader(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def to_categorical(y, num_classes=None, dtype='float32'):
    """
    From https://github.com/keras-team/keras/blob/master/keras/utils/np_utils.py

    :param y: input class labels as integers
    :param num_classes: how many classes
    :param dtype: numpy array data type
    :return: A binary matrix representation of the input. The classes axis is placed last.
    """
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


def accuracy(x, y):
    """
    Calculates accuracy, simple as that.

    :param x: predicted labels
    :param y: ground truth labels
    :return: accuracy
    """

    acc = np.sum(x == y).astype(np.float32) / x.shape[0]
    return acc


def set_train(net):
    """
    Sets a network to train mode (e.g. enable dropout), works inplace.

    :param net: the network to set
    :return: None
    """
    for layer in net:
        layer.training = True


def set_eval(net):
    """
    Sets a network to eval mode (e.g. disable dropout), works inplace.

    :param net: the network to set
    :return: None
    """
    for layer in net:
        layer.training = False


def check_grad_input(func, inputs, eps=1e-6):
    forward = func.forward(inputs)
    grad = func.backward(np.ones_like(forward))
    num_grad = ...  # todo: compute numerical jacobian, multiply by np.ones_like(forward)
    print(grad)
    print(num_grad)
    return np.allclose(grad, num_grad, atol=1e-5, rtol=1e-3), grad, num_grad
