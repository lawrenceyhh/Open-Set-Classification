import tensorflow as tf
import torch
import torch.nn as nn


class TF_COB_CNN(tf.keras.layers.Layer):
    def __init__(self, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        # self.conv1 = tf.keras.layers.Conv2D(32, (7, 7), padding='same', activation='relu', input_shape=(754, 27, 3))
        # self.maxpool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        #
        # self.conv2 = tf.keras.layers.Conv2D(64, (5, 5), activation='relu')
        # self.maxpool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        #
        # self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')
        # self.maxpool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        #
        # self.dropout1 = tf.keras.layers.Dropout(0.45)
        #
        # self.flatten = tf.keras.layers.Flatten()
        #
        # self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        # self.dropout2 = tf.keras.layers.Dropout(0.45)
        # self.dense2 = tf.keras.layers.Dense(self.num_classes)
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (7, 7), padding='same', activation='relu', input_shape=(754, 27, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Dropout(0.45),

            tf.keras.layers.Flatten(),

            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.45),
            tf.keras.layers.Dense(self.num_classes)
        ])

    def call(self, inputs):
        # x = self.conv1(inputs)
        # x = self.maxpool1(x)
        #
        # x = self.conv2(x)
        # x = self.maxpool2(x)
        #
        # x = self.conv3(x)
        # x = self.maxpool3(x)
        #
        # x = self.dropout1(x)
        #
        # x = self.flatten(x)
        #
        # x = self.dense1(x)
        # x = self.dropout2(x)
        # x = self.dense2(x)
        return self.model(inputs)


class COB_CNN(nn.Module):
    def __init__(self, num_classes=8):
        super(COB_CNN, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout2d(p=0.45)
        self.dropout2 = nn.Dropout(p=0.45)

        self.fc1 = nn.Linear(11776, 128)
        self.fc2 = nn.Linear(128, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)

        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        # x = nn.functional.softmax(x, dim=1)

        return x


# these two models below are deprecated 
class COB_CNN_9(nn.Module):
    def __init__(self):
        super(COB_CNN_9, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout2d(p=0.45)
        self.dropout2 = nn.Dropout(p=0.45)

        self.fc1 = nn.Linear(11776, 128)
        self.fc2 = nn.Linear(128, 9)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)

        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        # x = nn.functional.softmax(x, dim=1)

        return x


class COB_CNN_8(nn.Module):
    def __init__(self):
        super(COB_CNN_8, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)

        self.dropout1 = nn.Dropout2d(p=0.45)
        self.dropout2 = nn.Dropout(p=0.45)

        self.fc1 = nn.Linear(11776, 128)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = nn.functional.relu(x)
        x = self.pool3(x)

        x = self.dropout1(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)
        # x = nn.functional.softmax(x, dim=1)

        return x
