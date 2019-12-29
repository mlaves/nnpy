import numpy as np
import nnpy
from keras.datasets import mnist

# create network
net = [
    nnpy.Flatten(keep_channels=True),
    nnpy.Linear(28*28, 128),
    nnpy.ReLU(),
    nnpy.BatchNorm(1),
    nnpy.Linear(128, 64),
    nnpy.Sigmoid(),
    nnpy.Flatten(),
    nnpy.Linear(64, 10)
]

# create loss function
criterion = nnpy.CrossEntropy()

# create optimizer
sgd = nnpy.SGD(net, learn_rate=0.01, momentum=0.01)

# create data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize
X_train = X_train.astype(np.float32) / 255
X_test = X_test.astype(np.float32) / 255

#y_train = nnpy.to_categorical(y_train, num_classes=10)
#y_test = nnpy.to_categorical(y_test, num_classes=10)


# make 4d
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)


def train():
    epoch_loss = []
    epoch_predictions = []
    epoch_labels = []

    nnpy.set_train(net)

    for x_batch, y_batch in nnpy.dataloader(X_train, y_train, batchsize=128, shuffle=True):
        logits = nnpy.forward(net, x_batch)
        loss_value = criterion.forward(logits, y_batch).mean()
        nnpy.backward(net, criterion)
        sgd.step()

        epoch_loss.append(loss_value)
        epoch_predictions.append(np.argmax(logits, axis=-1))
        # epoch_labels.append(np.argmax(y_batch, axis=-1))
        epoch_labels.append(y_batch)

    epoch_predictions = np.array(epoch_predictions).flatten()
    epoch_labels = np.array(epoch_labels).flatten()

    return epoch_loss, epoch_predictions, epoch_labels


def test():
    epoch_loss = []
    epoch_predictions = []
    epoch_labels = []

    nnpy.set_eval(net)

    for x_batch, y_batch in nnpy.dataloader(X_test, y_test, batchsize=128, shuffle=False):
        logits = nnpy.forward(net, x_batch)
        loss_value = criterion.forward(logits, y_batch).mean()
        nnpy.backward(net, criterion)
        sgd.step()

        epoch_loss.append(loss_value)
        epoch_predictions.append(np.argmax(logits, axis=-1))
        # epoch_labels.append(np.argmax(y_batch, axis=-1))
        epoch_labels.append(y_batch)

    epoch_predictions = np.array(epoch_predictions).flatten()
    epoch_labels = np.array(epoch_labels).flatten()

    return epoch_loss, epoch_predictions, epoch_labels


epochs = 10
for e in range(epochs):

    epoch_loss_train, epoch_predictions_train, epoch_labels_train = train()
    epoch_loss_test, epoch_predictions_test, epoch_labels_test = test()

    print(f"epoch: {e:3d},",
          f"train loss: {np.mean(epoch_loss_train):.4f},",
          f"train accuracy: {nnpy.accuracy(epoch_predictions_train, epoch_labels_train):.4f},",
          f"test loss: {np.mean(epoch_loss_test):.4f},",
          f"test accuracy: {nnpy.accuracy(epoch_predictions_test, epoch_labels_test):.4f}"
          )
