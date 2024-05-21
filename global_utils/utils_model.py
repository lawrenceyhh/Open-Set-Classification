import tensorflow as tf
from tensorflow.keras import layers, activations


def sep_last_activation(model):
    # just a function to seperate the activation layer from the last dense layer
    model = remove_last_activation(model)
    model.add(layers.Activation(activations.softmax))
    return model


def remove_last_activation(model):
    # just a function to remove the activation function for the last dense layer
    model.layers[-1].activation = activations.linear
    return model
