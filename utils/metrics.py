import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
            np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (
        pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01*(u / d).mean(-1)


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


def metric(pred: np.ndarray, true: np.ndarray) -> dict:
    """
    Compute various evaluation metrics for
    predictions against ground truth values.

    Parameters
    ----------
    pred : np.ndarray
        Predicted values, typically of shape (n_samples, ...).
    true : np.ndarray
        Ground truth values, typically of shape (n_samples, ...).

    Returns
    -------
    dict
        A dictionary containing the following metrics:
        - 'mae' : Mean Absolute Error (MAE).
        - 'mse' : Mean Squared Error (MSE).
        - 'rmse' : Root Mean Squared Error (RMSE).
        - 'mape' : Mean Absolute Percentage Error (MAPE).
        - 'mspe' : Mean Squared Percentage Error (MSPE).
        - 'rse' : Root Square Error (RSE),
          normalised by the variance of `true`.
        - 'corr' : Correlation coefficient between `pred` and `true`.

    Notes
    -----
    - The `corr` (correlation coefficient) metric adds a small constant (1e-12)
      to the denominator to avoid division by zero.
    - If `true` contains zero values, MAPE and MSPE might
      produce undefined or infinite results.
    """
    if pred.shape != true.shape:
        raise ValueError(
            'the shape of pred and true must be the same, got'
            f' {pred.shape} and {true.shape}')

    metrics = {
        'mae' : MAE(pred, true),
        'mse' : MSE(pred, true),
        'rmse' : RMSE(pred, true),
        'mape' : MAPE(pred, true),
        'mspe' : MSPE(pred, true),
        'rse' : RSE(pred, true),
        'corr' : CORR(pred, true)
    }

    return metrics
