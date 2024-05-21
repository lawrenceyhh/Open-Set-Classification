from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List
import os
import numpy as np
import cv2
import copy
import random


class MyDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform is not None:
            x = copy.copy(self.data[index])
            x = self.transform(x)

        return x, y


def perturb_images(images, model, epsilon=0.005):
    # test_ds_var = tf.Variable(images, trainable=True)
    test_ds_var = images

    with tf.GradientTape() as tape:
        # Calculate the scores.
        tape.watch(test_ds_var)
        logits = model(test_ds_var, training=False)
        loss = tf.reduce_max(logits, axis=1)
        loss = -tf.reduce_mean(loss)

    # Calculate the gradients of the scores with respect to the inputs.
    gradients = tape.gradient(loss, test_ds_var)
    gradients = tf.math.greater_equal(gradients, 0)
    gradients = tf.cast(gradients, tf.float32)
    gradients = (gradients - 0.5) * 2

    # Perturb the inputs and derive new mean score.
    # test_ds_var.assign_add(epsilon * gradients)
    static_tensor = tf.convert_to_tensor(test_ds_var)
    static_tensor = static_tensor - epsilon * gradients
    static_tensor = tf.clip_by_value(static_tensor, 0., 255.)
    return static_tensor


def create_perturbed_tf_loader(tf_loader, model, epsilon):
    num_samples = tf.data.experimental.cardinality(tf_loader).numpy()
    perturbed_images = np.zeros((num_samples, 754, 27, 3))
    unchanged_labels = np.zeros((num_samples,))

    for i, (image, label) in enumerate(tqdm(tf_loader)):
        # image = tf.expand_dims(image, 0)
        perturbed_images[i] = perturb_images(image, model, epsilon)
        unchanged_labels[i] = label

    perturbed_ds = tf.data.Dataset.from_tensor_slices((perturbed_images, unchanged_labels))
    return perturbed_ds


def create_tf_loader(x_data, y_data, batch_size=128, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((tf.cast(x_data, tf.float32), y_data.argmax(1)))
    num_samples = tf.data.experimental.cardinality(ds).numpy()
    unchanged_images = np.zeros((num_samples, 754, 27, 3))
    unchanged_labels = np.zeros((num_samples,))
    for i, (image, label) in enumerate(tqdm(ds)):
        image = tf.expand_dims(image, 0)
        unchanged_images[i] = image
        unchanged_labels[i] = label
    data = tf.data.Dataset.from_tensor_slices((unchanged_images, unchanged_labels))
    if shuffle:
        return data.shuffle(buffer_size=batch_size).batch(batch_size)
    return data.batch(batch_size)


def create_torch_loader(x_data, y_data, batch_size=128, shuffle=False, transform=None):
    images = torch.from_numpy(x_data.transpose(0, 3, 1, 2))
    labels = torch.from_numpy(y_data.argmax(1))
    dataset = MyDataset(images, labels, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_ood_dataset(
        new_pattern_path: str,
        old_pattern_train_path: str,
        old_pattern_test_path: str,
        new_behavior_list: List[str] = ['NewPattern1'],
        old_behavior_list: List[str] = None,
        rescale=True, num_samples=300, to_split=True, seed=None):
    """
    Helper function to load the Out-of-Distribution dataset -> old patterns come first

    Args:
        new_pattern_path (str): Path to the new pattern dir
        old_pattern_train_path (str): Path to the old pattern train dir
        old_pattern_test_path (str): Path to the old pattern test dir
        new_behavior_list (list): List of new patterns to load. Defaults to ['NewPattern1'].
        old_behavior_list (list): List of old patterns to load. Defaults to None.
        rescale (bool, optional): Set True to rescale the images when loading. Defaults to True.
        num_samples (int, optional): Number of samples to sample from the new behavior. Defaults to 300.
        to_split (bool, optional): Set True to do the train-val-split. Defaults to True.
        seed (int, optional): Set seed for the random state in train_val_split

    Returns:
        X_ood, (X_ood_val), X_ood_test, y_ood, (y_ood_val), y_ood_test: The OOD dataset
    """
    X_OoD_all_new, y_OoD_all_new = load_data(new_pattern_path, new_behavior_list, rescale=rescale)

    # sample a number of samples from the new_pattern set for testing
    sample_index = random.sample(range(X_OoD_all_new.shape[0]), k=num_samples)
    X_OoD_test_new, y_OoD_test_new = X_OoD_all_new[sample_index], y_OoD_all_new[sample_index]

    # the rest for training and validation (new_pattern training/validation set)
    X_OoD_train_new = np.delete(X_OoD_all_new, sample_index, axis=0)
    y_OoD_train_new = np.delete(y_OoD_all_new, sample_index, axis=0)

    if old_behavior_list is None:
        if to_split:
            X_ood, X_ood_val, y_ood, y_ood_val = train_valid_split(
                X_OoD_train_new, y_OoD_train_new, test_size=0.2, seed=42)
            return X_ood, X_ood_val, X_OoD_test_new, y_ood, y_ood_val, y_OoD_test_new
        return X_OoD_train_new, X_OoD_test_new, y_OoD_train_new, y_OoD_test_new

    # join the old_pattern training/validation set w/ the new_pattern training/validation set
    X_OoD_test_old, y_OoD_test_old = load_data(old_pattern_test_path, old_behavior_list, rescale=rescale)
    X_OoD_train_old, y_OoD_train_old = load_data(old_pattern_train_path, old_behavior_list, rescale=rescale)
    X_ood = np.concatenate((X_OoD_train_old, X_OoD_train_new), axis=0)
    y_ood = np.zeros(
        (y_OoD_train_old.shape[0] + y_OoD_train_new.shape[0], y_OoD_train_old.shape[1] + y_OoD_train_new.shape[1]))
    y_ood[:y_OoD_train_old.shape[0], :y_OoD_train_old.shape[1]] = y_OoD_train_old
    y_ood[y_OoD_train_old.shape[0]:, y_OoD_train_old.shape[1]:] = y_OoD_train_new

    # join the old_pattern testing set w/ the new_pattern testing set
    X_ood_test = np.concatenate((X_OoD_test_old, X_OoD_test_new), axis=0)
    y_ood_test = np.zeros(
        (y_OoD_test_old.shape[0] + y_OoD_test_new.shape[0], y_OoD_test_old.shape[1] + y_OoD_test_new.shape[1]))
    y_ood_test[:y_OoD_test_old.shape[0], :y_OoD_test_old.shape[1]] = y_OoD_test_old
    y_ood_test[y_OoD_test_old.shape[0]:, y_OoD_test_old.shape[1]:] = y_OoD_test_new

    if to_split:
        # if needed, do the training and validation split on the joined dataset
        X_ood, X_ood_val, y_ood, y_ood_val = train_valid_split(X_ood, y_ood, test_size=0.2, seed=seed)
        return X_ood, X_ood_val, X_ood_test, y_ood, y_ood_val, y_ood_test
    return X_ood, X_ood_test, y_ood, y_ood_test


def load_data(data_path, behavior_list=None, rescale=True):
    #     val_dir_list = sorted(os.listdir(data_path))
    #     labels_name={'Constant':0, 'DoubleBooking':1, 'DownUp':2, 'DownUpDown':3, 'Overplanning':4, 'Random':5,
    #                  'TenTimesBooking':6, 'Underplanning':7, 'UpDown':8, 'UpDownUp':9}
    # if behavior list is not specified -> use all behaviors
    if behavior_list is None:
        behavior_list = [
            'Constant', 'DoubleBooking', 'DownUp', 'DownUpDown', 'Overplanning',
            'Random', 'TenTimesBooking', 'Underplanning', 'UpDown', 'UpDownUp']
    num_classes = len(behavior_list)
    labels_name = {behavior_list[i]: i for i in range(num_classes)}

    image_list = []
    label_list = []

    print(f'loading behaviors: {behavior_list}')
    for data_class in tqdm(behavior_list, position=0, leave=True):
        file_list = os.listdir(data_path + '/' + data_class)
        #         print ('Loaded the images of data_class-'+'{}\n'.format(data_class))
        label = labels_name[data_class]
        for file in file_list:
            input_image = cv2.imread(data_path + '/' + data_class + '/' + file)
            input_image = cv2.cvtColor(input_image, cv2.COLORMAP_RAINBOW)
            input_image_resize = cv2.resize(input_image, (27, 754))
            image_list.append(input_image_resize)
            label_list.append(label)

    image_data = np.array(image_list)
    image_data = image_data.astype('float32')
    if rescale:
        image_data /= 255.
    print(image_data.shape)
    labels = np.array(label_list)
    Y = np_utils.to_categorical(labels, num_classes)
    X_data, y_data = (image_data, Y)
    print(f'{labels_name}\n\n')
    return X_data, y_data


def train_valid_split(X_data, y_data, test_size=0.2, seed=None):
    train_X, train_y = [], []
    valid_X, valid_y = [], []

    for i in range(y_data.shape[1]):
        X_train, X_val, y_train, y_val = train_test_split(
            X_data[(i == y_data.argmax(1))],
            y_data[(i == y_data.argmax(1))],
            test_size=test_size,
            random_state=seed)
        train_X.append(X_train)
        train_y.append(y_train)
        valid_X.append(X_val)
        valid_y.append(y_val)

    train_X = np.concatenate(train_X, axis=0)
    valid_X = np.concatenate(valid_X, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    valid_y = np.concatenate(valid_y, axis=0)
    return train_X, valid_X, train_y, valid_y


def add_unknown_data(data_path, new_label, rescale=True):
    all_image = os.listdir(os.path.join(data_path, new_label))
    input_list = []
    for image_name in all_image:
        input_img = cv2.imread(os.path.join(data_path, new_label, image_name))
        input_img = cv2.cvtColor(input_img, cv2.COLORMAP_RAINBOW)
        input_resize = cv2.resize(input_img, (27, 754))
        input_list.append(input_resize)
    X_data = np.array(input_list)
    X_data = X_data.astype('float32')
    if rescale:
        X_data /= 255
    print(X_data.shape)
    return X_data


def join_unknown_dataset(data_class, label1, dataset2):
    X_joined = np.concatenate([data_class, dataset2], axis=0)
    y_joined = np.zeros((data_class.shape[0] + dataset2.shape[0], label1.shape[1] + 1))
    y_joined[:label1.shape[0], :label1.shape[1]] = label1
    y_joined[label1.shape[0]:, -1] = 1
    return X_joined, y_joined


def join_unknown_dataset_same_size(data_class, label1, dataset2):
    X_joined = np.concatenate([data_class, dataset2], axis=0)
    y_joined = np.zeros((data_class.shape[0] + dataset2.shape[0], label1.shape[1]))
    y_joined[:label1.shape[0], :label1.shape[1]] = label1
    y_joined[label1.shape[0]:, -1] = 1
    return X_joined, y_joined
