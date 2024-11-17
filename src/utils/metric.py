from sklearn.metrics import f1_score, precision_score, recall_score


def evaluate_metrics(true_labels, predicted_labels, average='weighted'):
    return {
        'f1': f1_score(true_labels, predicted_labels, average=average),
        'precision': precision_score(true_labels, predicted_labels, average=average),
        'recall': recall_score(true_labels, predicted_labels, average=average),
    }
