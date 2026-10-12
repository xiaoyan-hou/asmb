from sklearn.metrics import roc_auc_score
def image_auroc(scores, labels): return float(roc_auc_score(labels, scores)) if len(set(labels)) > 1 else float("nan")
