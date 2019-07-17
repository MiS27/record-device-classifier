import lasagne
import numpy as np


def build_crnn(input_var=None, height=None, width=None):
    example = np.random.uniform(size=(10, 1, 35, 129), low=0.0, high=1.0).astype(np.float32)  #########
    network = lasagne.layers.InputLayer(shape=(None, 1, height, width),
                                        input_var=input_var)
    print(lasagne.layers.get_output(network).eval({input_var: example}).shape)
    network = lasagne.layers.Conv2DLayer(
        network,
        num_filters=16, filter_size=(3, 3), pad='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    print(lasagne.layers.get_output(network).eval({input_var: example}).shape)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 2))
    print(lasagne.layers.get_output(network).eval({input_var: example}).shape)
    network = lasagne.layers.Conv2DLayer(
        network,
        num_filters=16, filter_size=(3, 3), pad='same',
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(1, 2))
    network = lasagne.layers.DimshuffleLayer(network, (0, 3, 1, 2))
    network = lasagne.layers.FlattenLayer(network, 3)
    network = lasagne.layers.GRULayer(network, num_units=16, only_return_final=True)
    network = lasagne.layers.DenseLayer(network, num_units=16, nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(network, num_units=1, nonlinearity=lasagne.nonlinearities.sigmoid)

    params = lasagne.layers.get_all_params(network, trainable=True)

    return network, params
