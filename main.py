#!/usr/bin/env python

from __future__ import print_function

import time

import numpy as np
from sklearn.model_selection import KFold

from data_util import DataUtil
from model import Model

debug = True


def main(num_epochs=100, n_splits=5):
    data_util = DataUtil('data', 'spectrogram_data')
    X, y = data_util.get_data()
    kf = KFold(n_splits=n_splits, shuffle=True)
    test_accuracy_sum = 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = Model(data_util.height, data_util.width)

        param_values, threshold = train_and_validate(model, X_train, y_train, num_epochs)
        model.set_param_values(param_values)

        test_accuracy_sum += perform_validation(model, X_test, y_test, threshold)
    print("Cross-validation results:")
    print("  accuracy:\t\t{:.2f} %".format(test_accuracy_sum/n_splits * 100))


def train_and_validate(model, inputs, targets, num_epochs):
    indices = np.arange(len(inputs))
    np.random.shuffle(indices)
    inputs = inputs[indices]
    targets = targets[indices]
    X_validation = inputs[:200]
    y_validation = targets[:200]
    X_train = inputs[200:]
    y_train = targets[200:]

    best_params = train_and_choose_best_params(X_train, X_validation, model, num_epochs, y_train, y_validation)
    model.set_param_values(best_params)

    return best_params, choose_best_threshold(model, X_validation, y_validation)


def train_and_choose_best_params(X_train, X_validation, model, num_epochs, y_train, y_validation):
    best_accuracy = 0
    best_params = None
    counter = 0
    for epoch in range(num_epochs):
        counter += 1
        accuracy = perform_epoch(model, X_train, y_train, X_validation, y_validation, epoch, num_epochs)
        if accuracy > best_accuracy:
            counter = 0
            best_accuracy = accuracy
            best_params = model.get_param_values()
        elif counter > 20:
            break
    return best_params


def choose_best_threshold(model, inputs, targets):
    best_accuracy = 0
    best_threshold = 0
    for threshold in np.arange(0.05, 1, 0.05):
        accuracy = perform_validation(model, inputs, targets, threshold)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold


def perform_epoch(model, X_train, y_train, X_vaildation, y_validation, epoch, num_epochs):
    start_time = time.time()
    train_loss, train_accuracy = model.train(X_train, y_train)
    if debug:
        print("Epoch {}/{} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
        print("  train loss:\t\t{:.6f}".format(train_loss))
        print("  train accuracy:\t\t{:.2f} %".format(train_accuracy * 100))
    return perform_validation(model, X_vaildation, y_validation)


def perform_validation(model, inputs, targets, threshold=None):
    threshold = threshold or 0.5
    loss, accuracy = model.validate(inputs, targets, threshold)
    if debug:
        print("Validation results:")
        print("  loss:\t\t\t{:.6f}".format(loss))
        print("  accuracy:\t\t{:.2f} %".format(accuracy * 100))
    return accuracy


if __name__ == '__main__':
    main()
