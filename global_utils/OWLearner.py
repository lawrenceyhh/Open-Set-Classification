import numpy as np
from OSClassifiers import OSClassifier
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import kaleido
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd
from abc import abstractmethod
from itertools import combinations, permutations
from sklearn.cluster import AgglomerativeClustering, DBSCAN


def get_threshold_g_mean(tpr, fpr, thresholds) -> float:
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = round(thresholds[index], ndigits=4)
    gmeanOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)
    print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))
    return thresholdOpt


def get_threshold_youden_j(tpr, fpr, thresholds) -> float:
    youdenJ = tpr - fpr
    gmean = np.sqrt(tpr * (1 - fpr))
    # Find the optimal threshold
    index = np.argmax(youdenJ)
    thresholdOpt = round(thresholds[index], ndigits=4)
    youdenJOpt = round(gmean[index], ndigits=4)
    fprOpt = round(fpr[index], ndigits=4)
    tprOpt = round(tpr[index], ndigits=4)
    print('Best Threshold: {} with Youden J statistic: {}'.format(thresholdOpt, youdenJOpt))
    print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))
    return thresholdOpt


def plot_scatter_neg(all_neg_logits):
    fig = px.scatter_matrix(all_neg_logits, width=1200, height=1600)
    fig.show()


def _heap_perm_(n, A):
    # code referred from https://stackoverflow.com/questions/29042819/heaps-algorithm-permutation-generator and
    # https://sedgewick.io/wp-content/themes/sedgewick/talks/2002PermGeneration.pdf
    if n == 1:
        yield A
    else:
        for i in range(n - 1):
            for hp in _heap_perm_(n - 1, A): yield hp
            j = 0 if (n % 2) == 1 else i
            A[j], A[n - 1] = A[n - 1], A[j]
        for hp in _heap_perm_(n - 1, A): yield hp


def count_correct(pseudo_labels, real_labels):
    pseudo_classes = len(np.unique(pseudo_labels))
    real_classes = len(np.unique(real_labels))

    if pseudo_classes < real_classes:
        max_correct, pesu_real = count_correct(real_labels, pseudo_labels)
        real_pesu = {v: k for k, v in pesu_real.items()}
        return max_correct, real_pesu

    _pseudo_labels = pseudo_labels + real_classes

    max_correct = 0
    best_combi = {}

    for combo in combinations(list(np.unique(_pseudo_labels)), real_classes):
        for combo_fact in permutations(combo):
            correct = 0
            ori_pseudo_lab = np.array(combo_fact) - real_classes

            for i in range(real_classes):
                correct += np.count_nonzero(real_labels[np.where(pseudo_labels == ori_pseudo_lab[i])] == i)

            if correct > max_correct:
                max_correct = correct
                best_combi = {i: j for i, j in zip(range(real_classes), ori_pseudo_lab)}
    return max_correct, best_combi


class OWLearner:
    def __init__(self, ood_detector: OSClassifier):
        self.ood_detector = ood_detector
        self.ood_classifier = None
        self.logits_in = None
        self.logits_out = None
        self.all_neg_logits = None
        self.current_threshold = None

    def get_logits(self, loader):
        return self.ood_detector.cal_output(loader)

    def set_current_threshold(self, current_threshold):
        self.current_threshold = current_threshold

    def get_train_logits(self, id_train_loader, ood_train_loader):
        self.logits_in = self.get_logits(id_train_loader)
        self.logits_out = self.get_logits(ood_train_loader)
        confidence_in = tf.reduce_max(self.logits_in, 1).numpy()
        confidence_out = tf.reduce_max(self.logits_out, 1).numpy()
        self.all_neg_logits = np.concatenate(
            [self.logits_in[np.where(confidence_in < self.current_threshold)],
             self.logits_out[np.where(confidence_out < self.current_threshold)]], axis=0)
        print("Ready to train!")

    def cal_ood_idx(self, data_loader):
        logits = self.get_logits(data_loader)
        confidence = tf.reduce_max(logits, 1)
        return np.where(confidence < self.current_threshold)[0]

    @abstractmethod
    def classify_ood(self, ood_logits, **kwargs):
        pass

    def eval_ood_classifier(self, id_data_loader, ood_data_loader, to_print=True, use_training_data=True):
        if self.ood_classifier is None:
            print("Please train the ood classifier first by calling 'train_ood_classifier()'!")
            return
        logits_in_test = self.get_logits(id_data_loader)
        logits_out_test = self.get_logits(ood_data_loader)
        confidence_in_test = tf.reduce_max(logits_in_test, 1)
        confidence_out_test = tf.reduce_max(logits_out_test, 1)

        _label_id_test = np.concatenate([label.numpy() for _, label in id_data_loader], axis=0)
        label_id_test = _label_id_test[np.where(confidence_in_test >= self.current_threshold)]
        id_classification = logits_in_test[np.where(confidence_in_test >= self.current_threshold)].argmax(1)
        id_correct = np.count_nonzero(id_classification == label_id_test)

        _label_out_test = np.concatenate([label.numpy() for _, label in ood_data_loader], axis=0)
        label_out_test = _label_out_test[np.where(confidence_out_test < self.current_threshold)]
        ood_logits = logits_out_test[np.where(confidence_out_test < self.current_threshold)]
        ood_classification = self.classify_ood(ood_logits, use_training_data=use_training_data)
        # TODO: concat False Negatives to the OOD logits to do classification?

        # this part requires a heap's algo to permutate all possible combination of pseudo label and real label
        # ood_correct = np.count_nonzero(ood_classification == label_out_test)
        ood_correct, _ = count_correct(ood_classification, label_out_test)

        false_neg = _label_id_test[np.where(confidence_in_test < self.current_threshold)]
        false_pos = _label_out_test[np.where(confidence_out_test >= self.current_threshold)]

        # for true_label in range(2):
        #     for predict_label in range(2):
        #         label_count = np.count_nonzero(
        #             ood_classification[np.where(label_out_test == true_label)] == predict_label)
        #         total_count = len(ood_classification[np.where(label_out_test == true_label)])
        #         print(f"behavior: {true_label}, predict: {predict_label},\
        #                         percent_correct: {label_count/total_count}")
        result_dict = {
            "id_accuracy":        id_correct / len(label_id_test),
            "id_accuracy w/ fp":  id_correct / (len(label_id_test) + len(false_pos)),
            "ood_accuracy":       ood_correct / len(label_out_test),
            "ood_accuracy w/ fn": ood_correct / (len(label_out_test) + len(false_neg)),
            "total_avg_accuracy": (id_correct + ood_correct) / (len(label_id_test) + len(label_out_test))
        }
        if to_print:
            [print(f'{key: <19} = {value}') for key, value in result_dict.items()]
            # print(f"id_accuracy         = {id_correct / len(label_id_test)}")
            # print(f"id_accuracy w/ fp   = {id_correct / (len(label_id_test) + len(false_pos))}")
            # print(f"ood_accuracy        = {ood_correct / len(label_out_test)}")
            # print(f"ood_accuracy w/ fn  = {ood_correct / (len(label_out_test) + len(false_neg))}")
            # print(f"total_avg_accuracy  = {(id_correct + ood_correct) / (len(label_id_test) + len(label_out_test))}")
        return label_out_test, ood_classification, result_dict

    def predict(self, data_loader):
        logits = self.get_logits(data_loader)
        confidence = tf.reduce_max(logits, 1)
        id_index = np.where(confidence >= self.current_threshold)[0]
        ood_index = np.where(confidence < self.current_threshold)[0]
        id_classification = logits[id_index].argmax(1)
        ood_classification = self.ood_classifier.predict(logits[ood_index])
        return id_index, ood_index, id_classification, ood_classification

    @abstractmethod
    def train_ood_classifier(self, *args, **kwargs):
        pass


class KMeansLearner(OWLearner):
    def __init__(self, ood_detector: OSClassifier):
        super().__init__(ood_detector)

    def plot_elbow(self, to_scale=False):
        x = self.all_neg_logits
        if to_scale:
            scaler = MinMaxScaler()
            scaler.fit(x)
            x = scaler.transform(x)
        inertia = []
        for i in range(1, 11):
            kmeans = KMeans(
                n_clusters=i, init="k-means++",
                n_init=10,
                tol=1e-04, random_state=42)
            kmeans.fit(x)
            inertia.append(kmeans.inertia_)
        fig = go.Figure(data=go.Scatter(x=np.arange(1, 11), y=inertia))
        fig.update_layout(title="Inertia vs Cluster Number", xaxis=dict(range=[0, 11], title="Cluster Number"),
                          yaxis={'title': 'Inertia'})
        fig.show()

    def train_ood_classifier(self, *args, **kwargs):
        n_clusters = args[0]
        behaviors_list = args[1]
        to_save = kwargs.get('to_save') if kwargs.get('to_save') is not None else False
        x = self.all_neg_logits
        kmeans = KMeans(
            n_clusters=n_clusters, init="k-means++",
            n_init=10,
            tol=1e-04, random_state=42)
        kmeans.fit(x)
        clusters = pd.DataFrame(x, columns=behaviors_list)
        clusters['label'] = kmeans.labels_
        self.ood_classifier = kmeans
        polar = clusters.groupby("label").mean().reset_index()
        polar = pd.melt(polar, id_vars=["label"])
        fig4 = px.line_polar(polar, r="value", theta="variable", color="label", line_close=True, height=800, width=900)
        if to_save:
            pio.write_image(fig4, f"./Figures/OWR_K_means_{n_clusters}.png")
            # fig4.write_image(f"./Figures/OWR_K_means_{n_clusters}.png")
        fig4.show()

    def classify_ood(self, ood_logits, **kwargs):
        return self.ood_classifier.predict(ood_logits)


class HierarchicalClusteringLearner(OWLearner):
    def __init__(self, ood_detector: OSClassifier):
        super().__init__(ood_detector)

    def train_ood_classifier(self, *args, **kwargs):
        n_clusters = args[0]
        affinity = kwargs.get("affinity") if kwargs.get("affinity") is not None else 'euclidean'
        linkage = kwargs.get("linkage") if kwargs.get("linkage") is not None else 'complete'
        hierarchical_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity=affinity, linkage=linkage)
        self.ood_classifier = hierarchical_cluster

    def classify_ood(self, ood_logits, use_training_data=True):
        if not use_training_data:
            return self.ood_classifier.fit_predict(ood_logits)
        _all_logits = np.concatenate([self.all_neg_logits, ood_logits])
        _all_labels = self.ood_classifier.fit_predict(_all_logits)
        return _all_labels[-len(ood_logits):]


class DBSCANLearner(OWLearner):
    def __init__(self, ood_detector: OSClassifier):
        super().__init__(ood_detector)

    def train_ood_classifier(self, *args, **kwargs):
        eps = args[0]  # hyperparams
        min_samples = args[1]  # hyperparams
        metricstr = kwargs.get("metricstr") if kwargs.get("metricstr") is not None else 'euclidean'
        dbscan = DBSCAN(eps, min_samples, metricstr)
        self.ood_classifier = dbscan

    def classify_ood(self, ood_logits, use_training_data=True):
        if not use_training_data:
            return self.ood_classifier.fit_predict(ood_logits)
        _all_logits = np.concatenate([self.all_neg_logits, ood_logits])
        _all_labels = self.ood_classifier.fit_predict(_all_logits)
        return _all_labels[-len(ood_logits):]
