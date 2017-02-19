import numpy as np
import math
import sklearn.datasets as ds, matplotlib.pyplot as plt


def create_data():
    np.random.seed(0)
    X, y = moon_data()
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    y = y.reshape(y.shape[0], 1)
    y_new = create_well_formed_label_matrix(y)
    return X, y_new

def create_well_formed_label_matrix(y):
    classes = np.unique(y).size
    new_matrix = np.zeros((y.shape[0],classes))
    for i in range(y.shape[0]):
        new_matrix[i,y[i]] = 1
    # print(y, new_matrix)
    return new_matrix

def moon_data():
    return ds.make_moons(200, noise=0.20)


def tanh_activation_function(h_layer):
    h_layer = np.tanh(h_layer)
    print("h_layer:", h_layer)

    return h_layer


def softmax_func(layer):
    layer = np.exp(layer)
    layer = layer / sum(layer)
    return layer


def calc_cross_entropy_instance(a, y):
    return (y * math.log(a) + (1-y)*math.log(1-a))
    


def cross_entropy_loss(out_act, labels):
    cross_entropy = 0.0
    for i in range(out_act.shape[0]):
        cross_entropy += calc_cross_entropy_instance(out_act[i],labels[i])
    return cross_entropy



if __name__ == '__main__':

    # Gather the data:
    data, labels = create_data()
    # print(data.shape, labels.shape)

    n_epochs = 2
    n_hidden = 5
    n_output = np.unique(labels).size
    W_i_h = np.random.rand(data.shape[1], n_hidden)
    W_h_o = np.random.rand(n_hidden, n_output)
    # print(W_i_h.shape, W_h_o.shape)

    for epoch in range(n_epochs):
        loss = 0.0
        total_samples = data.shape[0]
        for i in range(total_samples):
            input = data[i, :]

            # forward pass:
            h_pre_act = np.dot(input, W_i_h)
            h_act = tanh_activation_function(h_pre_act)

            out_pre_act = np.dot(h_act, W_h_o)
            out_act = softmax_func(out_pre_act)

            loss += cross_entropy_loss(out_act, labels[i])
        avg_loss = loss / total_samples
