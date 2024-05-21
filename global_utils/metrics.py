# Code referred from:
# https://github.com/pokaxpoka/deep_Mahalanobis_detector/blob/master/calculate_log.py
# https://github.com/sayakpaul/Generalized-ODIN-TF/blob/main/scripts/metrics.py

import tensorflow as tf
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import torch
import pandas as pd
# import OODDetectors


def calculate_metrics_pt(model, dataloader, device, direction="min"):
    true_labels = []
    predictions = []

    if direction == "min":
        outputs = model(dataloader).argmin(1)
    else:
        outputs = model(dataloader).argmax(1)

    predictions.extend(outputs)

    for _, labels in dataloader:
        true_labels.extend(labels.numpy())

    df = summarize_report(true_labels, predictions, model.model_name)

    return df


def calculate_metrics_tf(model, tf_dataloader):
    true_labels = []
    predictions = []

    outputs = model(tf_dataloader)
    predictions.extend(outputs.argmax(1))

    for _, labels in tf_dataloader:
        true_labels.extend(labels.numpy())

    df = summarize_report(true_labels, predictions, model.model_name)

    return df


def summarize_report(true_labels, predictions, name, behaviors_list=None):
    if behaviors_list is None:
        behaviors_list = ['Constant', 'DoubleBooking', 'DownUp', 'DownUpDown', 'Overplanning',
                          'TenTimesBooking', 'Underplanning', 'UpDown']
    true_labels_names = [behaviors_list[i.astype(np.int64)] for i in true_labels]
    predictions_names = [behaviors_list[i.astype(np.int64)] for i in predictions]

    report = classification_report(true_labels_names, predictions_names, zero_division=0, output_dict=True)
    df = pd.DataFrame(report).transpose()
    print(classification_report(true_labels_names, predictions_names, zero_division=0))

    matrix = confusion_matrix(true_labels_names, predictions_names)
    per_class_acc = matrix.diagonal() / matrix.sum(axis=1)
    per_class_acc_dict = {behaviors_list[i]: acc for i, acc in enumerate(per_class_acc)}

    df_acc = pd.DataFrame(per_class_acc_dict, index=[name])

    f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
    print(f'F1 Score: {f1}')
    return df, df_acc


def get_confidence(loader, model, stype=None, device=None):
    # if issubclass(type(model), OODDetectors.OODDetector):
    if model.is_oodd():
        output = model.cal_output(loader)
        return tf.reduce_max(output, 1).numpy()

    if stype == "generalized_odin":
        logits = model.predict(loader)
        confidence = tf.reduce_max(logits, 1)
    elif stype == "OSDN":
        act_vecs = model.get_model_outputs(loader)
        # scores = np.array([model.compute_openmax(x) for x in act_vecs])[:, :-1]
        scores = np.array([model.compute_openmax(x) for x in act_vecs])[:, 1:]
        confidence = tf.reduce_max(scores, 1)
    elif stype == "CNN_OE":
        output_list = []
        model.eval()
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                batch_output = model(inputs)
                output_list.append(batch_output)
        outputs = torch.cat(output_list, dim=0).numpy()
        confidence = tf.reduce_max(outputs, 1)
    elif stype == "CNN_CAC":
        output_list = []
        model.eval()
        with torch.no_grad():
            for inputs, _ in loader:
                inputs = inputs.to(device)
                batch_output = model(inputs)
                output_list.append(batch_output[1])
        outputs = torch.cat(output_list, dim=0).numpy()
        confidence = tf.reduce_max(outputs, 1)
    else:
        confidence = model.predict(loader)
        confidence = tf.reduce_max(confidence, 1)
    return confidence.numpy()


def calculate_auroc(in_loader, out_loader, model, stype, device=None):
    tp, fp = dict(), dict()
    tnr_at_tpr95 = dict()

    confidence_in = get_confidence(in_loader, model=model, stype=stype, device=device)
    confidence_out = get_confidence(out_loader, model=model, stype=stype, device=device)

    confidence_in.sort()
    confidence_out.sort()

    thres = np.concatenate([confidence_in, confidence_out], axis=0)
    thres.sort()

    # end = np.max([np.max(confidence_in), np.max(confidence_out)])
    # start = np.min([np.min(confidence_in), np.min(confidence_out)])

    num_k = confidence_in.shape[0]
    num_n = confidence_out.shape[0]
    tp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
    fp[stype] = -np.ones([num_k + num_n + 1], dtype=int)
    tp[stype][0], fp[stype][0] = num_k, num_n

    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[stype][l + 1:] = tp[stype][l]
            fp[stype][l + 1:] = np.arange(fp[stype][l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[stype][l + 1:] = np.arange(tp[stype][l] - 1, -1, -1)
            fp[stype][l + 1:] = fp[stype][l]
            break
        else:
            if confidence_out[n] < confidence_in[k]:
                n += 1
                tp[stype][l + 1] = tp[stype][l]
                fp[stype][l + 1] = fp[stype][l] - 1
            else:
                k += 1
                tp[stype][l + 1] = tp[stype][l] - 1
                fp[stype][l + 1] = fp[stype][l]
        # print(f"l: {l}, k: {k}, n: {n}")
    tpr95_pos = np.abs(tp[stype] / num_k - 0.95).argmin()
    tnr_at_tpr95[stype] = 1.0 - fp[stype][tpr95_pos] / num_n

    return tp, fp, tnr_at_tpr95, thres


def metric(in_loader, out_loader, model, stype, device=None):
    tp, fp, tnr_at_tpr95, thres = calculate_auroc(in_loader, out_loader, model, stype, device)

    results = dict()
    results[stype] = dict()

    # TNR
    mtype = "TNR"
    results[stype][mtype] = tnr_at_tpr95[stype]

    # AUROC
    mtype = "AUROC"
    tpr = np.concatenate([[1.0], tp[stype] / tp[stype][0], [0.0]])
    fpr = np.concatenate([[1.0], fp[stype] / fp[stype][0], [0.0]])
    results[stype][mtype] = -np.trapz(1.0 - fpr, tpr)
    results[stype]['tpr'] = tpr
    results[stype]['fpr'] = fpr
    results[stype]['thres'] = thres

    return results
