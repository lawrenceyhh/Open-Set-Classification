import tensorflow as tf
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score
from abc import abstractmethod
import metrics as m

scce = tf.keras.losses.SparseCategoricalCrossentropy()


class OSClassifier:
    """This is a wrapper class for all the Out-of-Distribution Detection Models
    All subclasses of this class needs to implement the cal_output, cal_accuracy, cal_loss functions
    
    The pre-trained model loaded can be either a TensorFlow/Keras or PyTorch model. However, please ensure that
    the data_loader passed in each function is also of the same type as the loaded model to ensure compatibility
    and proper functioning.
    """
    def __init__(self, model, model_name, *args):
        """Parameters

        Args:
            model (TensorFlow/Keras(.h5) or PyTorch(.pt) model): the pre-trained model
            model_name (str): name of the pre-trained model
        """
        self.model = model
        self.model_name = model_name
        self.accuracy = None
        self.loss = None
        self.tnr_at_tpr95 = None
        self.auroc = None
        self.result_metric = None
        self.save_value = False

    def __call__(self, data_loader, *args, **kwargs):
        return self.cal_output(data_loader)

    @staticmethod
    def is_oodd():
        # this is a method to check for subclasses without using issubclass() method which doesn't work sometimes
        return True

    @abstractmethod
    def cal_output(self, data_loader):
        pass

    @abstractmethod
    def cal_accuracy(self, data_loader):
        pass

    @abstractmethod
    def cal_loss(self, data_loader):
        pass

    def cal_result_metric(self, id_data_loader, ood_data_loader):
        return m.metric(id_data_loader, ood_data_loader, self, self.model_name)

    def get_model_name(self):
        return self.model_name

    def get_accuracy(self, data_loader):
        if self.accuracy is None:
            self.accuracy = self.cal_accuracy(data_loader)
        return self.accuracy

    def get_loss(self, data_loader):
        if self.loss is None:
            self.loss = self.cal_loss(data_loader)
        return self.loss

    def get_result_metric(self, id_data_loader=None, ood_data_loader=None):
        if self.result_metric is not None:
            return self.result_metric
        if id_data_loader is None or ood_data_loader is None:
            raise ValueError("Data Loaders cannot be None!")
        self.result_metric = self.cal_result_metric(id_data_loader, ood_data_loader)
        self.tnr_at_tpr95 = self.result_metric.get(self.model_name).get('TNR')
        self.auroc = self.result_metric.get(self.model_name).get('AUROC')
        return self.result_metric

    def set_accuracy(self, accuracy):
        self.accuracy = accuracy

    def set_loss(self, loss):
        self.loss = loss

    def set_result_metric(self, metric):
        self.result_metric = metric

    def get_result(self, data_name, id_data_loader, ood_data_loader):
        result_metric = self.get_result_metric(id_data_loader, ood_data_loader)
        return {
            'Model': self.model_name,
            'Data': data_name,
            'Accuracy': self.get_accuracy(id_data_loader),
            'Loss': self.get_loss(id_data_loader),
            'TNR': result_metric.get(self.model_name).get('TNR'),
            'AUROC': result_metric.get(self.model_name).get('AUROC'),
        }, result_metric


class BaselineOSC(OSClassifier):
    def __init__(self, model, model_name):
        super().__init__(model, model_name)

    def cal_output(self, data_loader):
        return self.model.predict(data_loader)

    def cal_accuracy(self, data_loader):
        _, accuracy = self.model.evaluate(data_loader)
        return accuracy

    def cal_loss(self, data_loader):
        _label = tf.concat([label for _, label in data_loader], axis=0)
        _out = self.cal_output(data_loader)
        return scce(_label, _out).numpy()


class GenODIN(BaselineOSC):
    def __init__(self, model, model_name):
        super().__init__(model, model_name)


class OpenSetDeepNetwork(OSClassifier):
    def __init__(self, model, model_name, *args):
        super().__init__(model, model_name)
        x_train = args[0]
        y_train = args[1]
        self.model.compute_class_means(x_train, y_train)

    def cal_output(self, data_loader):
        act_vecs = self.model.get_model_outputs(data_loader)
        scores = np.array([self.model.compute_openmax(x) for x in act_vecs])[:, :-1]
        return scores

    def cal_accuracy(self, data_loader):
        scores = self.cal_output(data_loader)
        _label = tf.concat([label for _, label in data_loader], axis=0).numpy()
        return accuracy_score(scores.argmax(1), _label)

    def cal_loss(self, data_loader):
        _label = tf.concat([label for _, label in data_loader], axis=0)
        _out = self.cal_output(data_loader)
        return scce(_label, _out).numpy()


class ClassAnchorClusteringModel(OSClassifier):
    def __init__(self, model, model_name, *args):
        super().__init__(model, model_name)
        self.device = args[0]
        self.model.to(self.device)

    def cal_output(self, data_loader):
        output_list = []
        self.model.eval()
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(self.device)
                batch_output = self.model(inputs)
                output_list.append(batch_output[1])
        outputs = torch.cat(output_list, dim=0)
        if self.device.type == 'cuda':
            return outputs.cpu().numpy()
        return outputs.numpy()

    def cal_accuracy(self, data_loader):
        self.model.eval()
        correct = 0
        correct_two = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)

                _, predicted = outputs[1].min(1)
                _, predicted_two = outputs[0].max(1)

                total += targets.size(0)

                correct += predicted.eq(targets).sum().item()
                correct_two += predicted_two.eq(targets).sum().item()
                # the two accuracy would be identical
        return max(correct / total, correct_two / total)

    def cal_loss(self, data_loader):
        _out = torch.from_numpy(self.cal_output(data_loader))
        _out = F.softmax(_out, dim=1).numpy()
        _label = torch.cat([label for _, label in data_loader], axis=0).numpy()
        return scce(_label, _out).numpy()


class OutlierExposureModel(OSClassifier):
    def __init__(self, model, model_name, *args):
        super().__init__(model, model_name)
        self.device = args[0]
        self.model.to(self.device)

    def cal_output(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            for index, (data, _) in enumerate(data_loader):
                data = data.to(self.device)
                # forward
                output = self.model(data)
                # accuracy
                if index == 0:
                    # initialize all_output
                    all_output = output
                    continue
                all_output = torch.cat((all_output, output), 0)
        if self.device.type == 'cuda':
            return all_output.cpu().numpy()
        return all_output.numpy()

    def cal_accuracy_and_loss(self, data_loader):
        self.model.eval()
        loss_avg = 0.0
        correct = 0
        total_len = len(data_loader.dataset)
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)

                # forward
                output = self.model(data)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        avg_loss = loss_avg / total_len
        avg_accuracy = correct / total_len
        self.loss = avg_loss
        self.accuracy = avg_accuracy

    def cal_accuracy(self, data_loader):
        if self.accuracy is None:
            self.cal_accuracy_and_loss(data_loader)
        return self.accuracy

    def cal_loss(self, data_loader):
        if self.loss is None:
            self.cal_accuracy_and_loss(data_loader)
        return self.loss
