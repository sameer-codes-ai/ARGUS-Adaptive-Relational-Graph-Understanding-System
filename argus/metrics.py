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

    if len(p) != len(y) or len(p) == 0:
        raise ValueError("predictions and targets must be non-empty and have equal length")
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError("predictions and targets must contain only finite values")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("targets must be binary values 0 or 1")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

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
        auc_roc = float('nan')
        ap = float('nan')
    else:
        # Advance the ROC/PR curve once per distinct score so tied predictions
        # cannot change the metric when input rows are reordered.
        sorted_scores = p[sorted_indices]
        distinct_ends = np.r_[np.flatnonzero(np.diff(sorted_scores)) + 1, len(sorted_scores)]
        tp_cumsum = np.cumsum(y_sorted)[distinct_ends - 1]
        fp_cumsum = np.cumsum(1 - y_sorted)[distinct_ends - 1]
        tpr = np.r_[0.0, tp_cumsum / n_pos]
        fpr = np.r_[0.0, fp_cumsum / n_neg]
        trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
        auc_roc = float(trapz_fn(tpr, fpr))

        precisions = tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1)
        recalls = tp_cumsum / n_pos
        recall_steps = np.diff(np.r_[0.0, recalls])
        ap = float(np.sum(precisions * recall_steps))


    return {
        'accuracy': round(float(acc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'auc_roc': round(float(auc_roc), 4),
        'average_precision': round(float(ap), 4),
    }
