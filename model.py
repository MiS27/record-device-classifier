import lasagne
import theano
import theano.tensor as T
import numpy as np

from build_crnn import build_crnn


class Model:
    def __init__(self, height, width, batch_size=50, l2_rate=0.001, learning_rate=0.003):
        self.batch_size = batch_size

        input_var = T.tensor4('inputs')
        target_var = T.ivector('targets')
        threshold = T.scalar('threshold', dtype=theano.config.floatX)

        network, params = build_crnn(input_var, height, width)
        self.network = network

        prediction = lasagne.layers.get_output(network)

        l2 = lasagne.regularization.apply_penalty(params, lasagne.regularization.l2)
        loss = lasagne.objectives.binary_crossentropy(prediction, target_var)
        loss = loss.mean() + l2_rate * l2
        train_accuracy = lasagne.objectives.binary_accuracy(prediction, target_var, 0.5).mean()
        updates = lasagne.updates.rmsprop(loss, params, learning_rate=learning_rate)
        self._train_fn = theano.function([input_var, target_var], [loss, train_accuracy], updates=updates)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.binary_crossentropy(test_prediction, target_var)
        test_loss = test_loss
        test_loss = test_loss.mean()
        test_accuracy = lasagne.objectives.binary_accuracy(test_prediction, target_var, threshold)
        test_accuracy = test_accuracy.mean()
        self._validate_fn = theano.function([input_var, target_var, threshold], [test_loss, test_accuracy])

    def train(self, X_train, y_train):
        loss_sum = 0
        batches = 0
        accuracy_sum = 0
        for (inputs, targets) in self._iterate_minibatches(X_train, y_train, True):
            loss, accuracy = self._train_fn(inputs, targets)
            loss_sum += loss
            accuracy_sum += accuracy
            batches += 1
        loss = loss_sum / batches
        accuracy = accuracy_sum / batches
        return loss.mean(), accuracy.mean()

    def validate(self, inputs, targets, threshold):
        loss, accuracy = self._validate_fn(inputs, targets, threshold)
        return loss.mean(), accuracy.mean()

    def get_param_values(self):
        return lasagne.layers.get_all_param_values(self.network)

    def set_param_values(self, param_values):
        lasagne.layers.set_all_param_values(self.network, param_values)

    def _iterate_minibatches(self, inputs, targets, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - self.batch_size + 1, self.batch_size):
            if shuffle:
                excerpt = indices[start_idx:start_idx + self.batch_size]
            else:
                excerpt = slice(start_idx, start_idx + self.batch_size)
            yield inputs[excerpt], targets[excerpt]
