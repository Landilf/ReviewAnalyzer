from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(true_labels, pred_labels):
    return {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "precision": precision_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "recall": recall_score(true_labels, pred_labels, average="weighted", zero_division=0),
        "F1": f1_score(true_labels, pred_labels, average="weighted", zero_division=0),
    }