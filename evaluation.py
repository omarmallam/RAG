from sklearn.metrics import precision_score, recall_score, f1_score
from typing import List, Dict

# -------------- RETRIEVAL METRICS ------------------

def evaluate_retrieval(relevant_docs: List[str], retrieved_docs: List[str]) -> Dict[str, float]:
    """
    relevant_docs and retrieved_docs are lists of document IDs (or paths).
    """
    all_docs = list(set(relevant_docs + retrieved_docs))
    y_true = [1 if doc in relevant_docs else 0 for doc in all_docs]
    y_pred = [1 if doc in retrieved_docs else 0 for doc in all_docs]

    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }

# -------------- ANSWER QUALITY ----------------------

def evaluate_answers(predictions: List[str], ground_truths: List[str]) -> Dict[str, float]:
    """
    Evaluates answers using exact match accuracy.
    Could be replaced with fuzzy match or BLEU/ROUGE for more flexibility.
    """
    correct = sum([1 for pred, gt in zip(predictions, ground_truths) if pred.strip().lower() == gt.strip().lower()])
    total = len(ground_truths)

    return {
        "exact_match_accuracy": correct / total
    }

# -------------- CONFIGURATION COMPARISON ------------

def compare_configurations(results: Dict[str, Dict[str, float]]) -> None:
    """
    Accepts a dictionary like:
    {
        "Config A": {"precision": 0.8, "recall": 0.75, "f1": 0.77},
        "Config B": {"precision": 0.85, "recall": 0.65, "f1": 0.73},
    }
    Prints side-by-side comparison.
    """
    print("\nConfiguration Comparison:")
    print(f"{'Configuration':<15} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
    print("-" * 50)
    for config_name, metrics in results.items():
        print(f"{config_name:<15} {metrics['precision']:<10.2f} {metrics['recall']:<10.2f} {metrics['f1']:<10.2f}")
