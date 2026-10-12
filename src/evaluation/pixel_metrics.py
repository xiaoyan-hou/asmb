from sklearn.metrics import roc_auc_score
def pixel_auroc(scores, masks):
    y = masks.detach().cpu().numpy().reshape(-1); s = scores.detach().cpu().numpy().reshape(-1)
    return float(roc_auc_score(y, s)) if len(set(y.tolist())) > 1 else float("nan")
