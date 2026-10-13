import torch
import numpy as np

def binary_metrics(preds, targets, threshold=0.5):
    if isinstance(preds, torch.Tensor):
        p = preds.detach().cpu().numpy().flatten()
    else:
        p = np.asarray(preds).flatten()

    if isinstance(targets, torch.Tensor):
        y = targets.detach().cpu().numpy().flatten()
    else:
        y = np.asarray(targets).flatten()

    p_bin = (p >= threshold).astype(np.int64)
    y_int = y.astype(np.int64)

    tp = int(((p_bin == 1) & (y_int == 1)).sum())
    fp = int(((p_bin == 1) & (y_int == 0)).sum())
    fn = int(((p_bin == 0) & (y_int == 1)).sum())
    tn = int(((p_bin == 0) & (y_int == 0)).sum())

    total = max(len(y), 1)
    acc = (tp + tn) / total
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * (prec * rec) / max(prec + rec, 1e-8)

    sorted_indices = np.argsort(-p)
    y_sorted = y_int[sorted_indices]
    n_pos = int(y_int.sum())
    n_neg = int(len(y_int) - n_pos)

    if n_pos == 0 or n_neg == 0:
        auc_roc = 0.5
        ap = float(n_pos > 0)
    else:
        tp_cumsum = np.cumsum(y_sorted)
        fp_cumsum = np.cumsum(1 - y_sorted)
        tpr = tp_cumsum / n_pos
        fpr = fp_cumsum / n_neg
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        auc_roc = float(trapz_fn(tpr, fpr)) if trapz_fn else 0.5

        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)
        ap = float(np.sum(precisions * y_sorted) / n_pos)


    return {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'auc_roc': round(float(auc_roc), 4),
        'average_precision': round(float(ap), 4),
    }
