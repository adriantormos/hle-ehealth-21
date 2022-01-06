import numpy as np
from datasets import load_metric


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    results = load_metric("seqeval").compute(predictions=predictions, references=labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }