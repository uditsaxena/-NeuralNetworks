import math

import numpy as np
import sklearn.datasets as ds


def create_data():
    X, y = moon_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    y = y.reshape(y.shape[0], 1)
    y_new = create_well_formed_label_matrix(y)
    return X, y_new, y


def create_well_formed_label_matrix(y):
    classes = np.unique(y).size
    new_matrix = np.zeros((y.shape[0], classes))
    for i in range(y.shape[0]):
        new_matrix[i, y[i]] = 1
    # print(y, new_matrix)
    return new_matrix


def moon_data():
    return ds.make_moons(200, noise=0.20)


def tanh_activation_function(h_layer):
    h_layer = np.tanh(h_layer)
    # print("h_layer:", h_layer)

    return h_layer


def softmax_func(layer):
    layer = np.exp(layer)
    layer = layer / np.sum(layer)
    return layer


def calc_cross_entropy_instance(a, y):
    return (y * math.log(a))


def cross_entropy_loss(out_act, labels):
    cross_entropy = 0.0
    for i in range(out_act.shape[0]):
        cross_entropy += calc_cross_entropy_instance(out_act[i], labels[i])
    return cross_entropy


def predict(data, W_i_h, W_h_o, bias_h, bias_out, y):
    total_samples = data.shape[0]
    out_act = np.zeros((data.shape[0], n_output))
    for i in range(total_samples):
        input = data[i, :]

        h_pre_act = np.dot(input, W_i_h) + bias_h
        h_act = tanh_activation_function(h_pre_act)

        out_pre_act = np.dot(h_act, W_h_o) + bias_out
        out_act[i, :] = softmax_func(out_pre_act)
    # print(out_act.shape, out_act)
    shape = np.argmax(out_act, axis=1)
    # print(shape.shape, y.shape)
    equal = np.equal(np.squeeze(shape), np.squeeze(y[:,0]))
    # print(equal.shape)
    print(np.count_nonzero(equal) / data.shape[0])


def batch_sgd(n_epochs, learning_rate, data, W_i_h, W_h_o, bias_h, bias_out, y, labels):
    for epoch in range(n_epochs):
        per_epoch_loss = 0.0
        total_samples = data.shape[0]
        for i in range(total_samples):
            input = data[i, :]

            # forward pass:
            h_pre_act = np.dot(input, W_i_h) + bias_h
            h_act = tanh_activation_function(h_pre_act)

            out_pre_act = np.dot(h_act, W_h_o) + bias_out
            out_act = softmax_func(out_pre_act)

            loss = cross_entropy_loss(out_act, labels[i])
            per_epoch_loss += loss
            # print(loss)
            # backward pass
            delta3 = out_act - labels[i]
            delta3 = delta3.reshape(1, delta3.shape[0])
            delta2 = (1 - np.power(np.tanh(h_pre_act), 2)) * (np.dot(delta3, W_h_o.transpose()))
            h_act = h_act.reshape(1, h_act.shape[0])
            delW_h_o = np.dot(h_act.transpose(), delta3)
            delB_o = delta3
            input = input.reshape(1, input.shape[0])
            delW_i_h = np.dot(input.transpose(), delta2)
            delB_h = delta2

            W_h_o += (-1) * learning_rate * np.squeeze(delW_h_o)
            W_i_h += (-1) * learning_rate * np.squeeze(delW_i_h)
            bias_out += (-1) * learning_rate * np.squeeze(delB_o)
            bias_h += (-1) * learning_rate * np.squeeze(delB_h)

        avg_loss = (-1) * per_epoch_loss / total_samples
        predict(data, W_i_h, W_h_o, bias_h, bias_out, y)
        # print(avg_loss)


if __name__ == '__main__':
    # Gather the data:
    data, labels, y = create_data()
    # print(data.shape, labels.shape)

    n_epochs = 200
    n_hidden = 5
    learning_rate = 0.01
    n_output = np.unique(labels).size
    W_i_h = np.random.rand(data.shape[1], n_hidden)
    W_h_o = np.random.rand(n_hidden, n_output)
    print(W_i_h.shape, W_h_o.shape)
    bias_h = np.random.rand(n_hidden)
    bias_out = np.random.rand(n_output)
    batch_sgd(n_epochs, learning_rate, data, W_i_h, W_h_o, bias_h, bias_out, y, labels)
