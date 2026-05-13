import os
import numpy as np
import pandas as pd
from keras.utils import to_categorical

def load_signals_from_csv(x_csv_path: str) -> np.ndarray:
    """
    CSV enthält: pro Sample 3 Zeilen (3 Kanäle).
    Danach reshape zu (n_samples, timesteps, channels=3).
    """
    X = np.loadtxt(x_csv_path)  # shape: (n_samples*3, timesteps)
    X = X.reshape(-1, 3, X.shape[-1]).transpose(0, 2, 1)  # (N, T, 3)
    return X

def standardize_per_channel(X_train: np.ndarray, X_other: np.ndarray, mu=None, sigma=None):
    """
    Standardisierung pro Kanal über (Samples, Timesteps), getrennt für jeden Kanal.
    X: (N, T, C)
    mu,sigma: (C,)
    """
    if mu is None or sigma is None:
        mu = X_train.mean(axis=(0, 1))
        sigma = X_train.std(axis=(0, 1))
        sigma[sigma == 0] = 1.0

    Xn = (X_other - mu) / sigma
    return Xn, mu, sigma

def preprocess(
    XTrain: np.ndarray,
    YTrainInt: pd.Series,
    XTest: np.ndarray,
    n_classes: int = 4,
    val_ratio: float = 0.15,
    seed: int = 21,
    save_dir: str = "data",):
    
    os.makedirs(save_dir, exist_ok=True)

    # Shuffle
    rng = np.random.default_rng(seed)
    idx = np.arange(XTrain.shape[0])
    rng.shuffle(idx)
    XTrain = XTrain[idx]
    y_int = np.asarray(YTrainInt).reshape(-1)[idx]

    # One-Hot
    Y = to_categorical(y_int, num_classes=n_classes)

    # Split: train / val
    n = XTrain.shape[0]
    val_size = int(val_ratio * n)
    train_size = n - val_size

    X_tr = XTrain[:train_size]
    Y_tr = Y[:train_size]
    X_val = XTrain[train_size:]
    Y_val = Y[train_size:]

    # Standardize per channel (fit on train only)
    X_tr_s, mu, sigma = standardize_per_channel(X_tr, X_tr, None, None)
    X_val_s, _, _ = standardize_per_channel(X_tr, X_val, mu, sigma)
    X_test_s, _, _ = standardize_per_channel(X_tr, XTest, mu, sigma)

    # Speichern: preprocessed Test für spätere Abgabe-Prediction
    np.save(os.path.join(save_dir, "XTest_preprocessed.npy"), X_test_s)
    np.save(os.path.join(save_dir, "mu.npy"), mu)
    np.save(os.path.join(save_dir, "sigma.npy"), sigma)

    return X_tr_s, Y_tr, X_val_s, Y_val