# ## The code for the vanilla OSDN class, code here is referred from the implementation here:
# https://october25kim.github.io/paper/openset/2020/10/11/openmax-paper/

import torch
import tensorflow as tf
import numpy as np
from libNotMR import LibNotMR
from tqdm.notebook import tqdm


class Openmax_Model:
    def __init__(self, model, number_of_classes=10, tailsize=20) -> None:
        super().__init__()
        """ 
        The OSDN model with openmax layer at the very end, can also be used as a normal CNN
        """
        self.model = model
        self.class_means = None
        self.mr_models = None
        self.number_of_classes = number_of_classes
        self.tailsize = tailsize

    def compute_class_means(self, X_train, y_train, batch_size=128):
        """
        To initialized the class means (MAV) and mr_models used later to calculate the openmax result

        Args:
            X_train (_type_): training data
            y_train (_type_): label data
            batch_size (int, optional): Defaults to 128
        """
        train_data = tf.data.Dataset.from_tensor_slices(X_train).batch(batch_size)
        train_pred_scores = self.get_model_outputs(train_data, False)
        train_pred_simple = np.argmax(train_pred_scores, axis=1)
        train_y = np.argmax(y_train, axis=1)
        train_correct_actvec = train_pred_scores[np.where(train_y == train_pred_simple)[0]]
        train_correct_labels = train_y[np.where(train_y == train_pred_simple)[0]]

        dist_to_means = []
        mr_models, class_means = [], []
        for c in tqdm(np.unique(train_y)):
            class_act_vec = train_correct_actvec[np.where(train_correct_labels == c)[0], :]
            class_mean = class_act_vec.mean(axis=0)
            dist_to_mean = np.square(class_act_vec - class_mean).sum(axis=1)
            dist_to_mean = np.sort(dist_to_mean).astype(np.float64)
            dist_to_means.append(dist_to_mean)
            mr = LibNotMR(self.tailsize)
            mr.fit_high(torch.tensor(dist_to_mean))
            class_means.append(class_mean)
            mr_models.append(mr)
        self.mr_models = mr_models
        self.class_means = np.array(class_means)

    def get_model_outputs(self, dataset, prob=False):
        """
        To get the activation scores, set prob=False; to get normal softmax outputs, set prob=True

        Args:
            dataset (_type_): _description_
            prob (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: _description_
        """
        pred_scores = []
        for data in dataset:
            if isinstance(data, tuple):
                x, _ = data
            else:
                x = data

            model_outputs = self.model(x, training=False)
            if prob:
                model_outputs = tf.nn.softmax(model_outputs)
            pred_scores.append(model_outputs.numpy())
        pred_scores = np.concatenate(pred_scores, axis=0)
        return pred_scores

    def compute_openmax(self, actvec):
        dist_to_mean = np.square(actvec - self.class_means).sum(axis=1).astype(np.float64)
        scores = []
        for dist, mr in zip(dist_to_mean, self.mr_models):
            scores.append(mr.w_score(dist))
        scores = np.array(scores)
        w = 1 - scores
        rev_actvec = np.concatenate([
            w * actvec,
            [((1 - w) * actvec).sum()]])
        return np.exp(rev_actvec) / np.exp(rev_actvec).sum()

    def get_logits(self, actvec):
        dist_to_mean = np.square(actvec - self.class_means).sum(axis=1).astype(np.float64)
        scores = []
        for dist, mr in zip(dist_to_mean, self.mr_models):
            scores.append(mr.w_score(dist))
        scores = np.array(scores)
        w = 1 - scores
        return w * actvec

    def make_prediction(self, _scores, _T, thresholding=True):
        _scores = np.array([self.compute_openmax(x) for x in _scores])
        if thresholding:
            uncertain_idx = np.where(np.max(_scores, axis=1) < _T)[0]
            uncertain_vec = np.zeros((len(uncertain_idx), self.number_of_classes + 1))
            uncertain_vec[:, -1] = 1
            _scores[uncertain_idx] = uncertain_vec
        _labels = np.argmax(_scores, 1)
        return _labels
