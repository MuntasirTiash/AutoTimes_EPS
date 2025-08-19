import numpy as np
from sklearn.metrics import r2_score


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


# ------------------------------------------------------------------
# new metrics
# ------------------------------------------------------------------
def R2(pred, true):
    """
    Coefficient of determination using sklearn's implementation.
    Works for any shape that flattens to (n_samples,).
    """
    return r2_score(true.reshape(-1), pred.reshape(-1))

def kelly_R2(pred, true):
    """
    Custom 'Kelly R²':
        1 - Σ(y - ŷ)² / Σ(y)²
    """
    diff = np.sum(np.square(true - pred)) / np.sum(np.square(true))
    return 1.0 - diff

# ------------------------------------------------------------------
# master wrapper (returns 7 numbers)
# ------------------------------------------------------------------
def metric(pred, true):
    mae   = MAE(pred, true)
    mse   = MSE(pred, true)
    rmse  = RMSE(pred, true)
    mape  = MAPE(pred, true)
    mspe  = MSPE(pred, true)
    r2    = R2(pred, true)
    k_r2  = kelly_R2(pred, true)
    return mae, mse, rmse, mape, mspe, r2, k_r2
