import tensorflow as tf
from tensorflow.keras import Input, Model


class TF_COB_ODIN(tf.keras.layers.Layer):
    def __init__(self, num_classes=None, l2_weight_decay=None):
        super().__init__()
        inputs = Input(shape=(754, 27, 3))
        x = CNN_body(inputs)
        outputs = classifier(x, num_classes, l2_weight_decay)
        self.model = Model(inputs, outputs)

    def call(self, inputs):
        return self.model(inputs)

    def CNN_body(self, x):
        x = Conv2D(32, (7, 7), padding='same', activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (5, 5), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D(2, 2)(x)
        x = Dropout(0.45)(x)
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.45)(x)
        #     x = Dense(8, activation='linear')(x)
        return x

    def classifier(self, x, n_classes=None, weight_decay=1e-5):
        """Construct a Classifier
        x         : input into the classifier
        n_classes : number of classes
        """

        # Define the ODIN as specified in Section 3.1.1 of
        # https://arxiv.org/abs/2002.11297
        h = Dense(n_classes, kernel_initializer="he_normal")(x)

        g = Dense(1, kernel_regularizer=l2(weight_decay))(x)
        g = BatchNormalization()(g)
        g = Activation("sigmoid")(g)
        outputs = tf.math.divide(h, g)

        return outputs