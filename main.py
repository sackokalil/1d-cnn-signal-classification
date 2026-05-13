#%% imports
import os.path
import numpy as np
import pandas as pd

from src.preprocessing import preprocess, load_signals_from_csv
from src.model import build_cnn_1d
from src.train_model import train_model

XTrainCSVFilename = 'XTrainSignale.csv'
YTrainCSVFilename = 'YTrainSignale.csv'
XTestCSVFilename  = 'XTestSignale.csv'
TrainDataZipFilename = 'TrainDataSignale.zip'
TestDataZipFilename  = 'XTestSignale.zip'

url_prefix = url_prefix = "https://hs-bochum.sciebo.de/s/2DS5EFSrhJqUTvA/download?path=%2Fzu%20l%C3%B6schen%2FUebungenWS2021-22%2Fdata&files="

#%% load data (Hilfscode)
class DownloadError(Exception):
    def __init__(self, msg, url=None, filename=None):
        super().__init__(msg)
        self.url = url
        self.filename = filename

def download_and_unzip(filename):
    if not os.path.isfile(filename):
        from urllib.request import urlretrieve
        url = url_prefix + filename
        print(f'Downloading {filename} from {url}.')
        try:
            urlretrieve(url, filename)
        except Exception as e:
            raise DownloadError('Could not Download the file', url, filename) from e

    print(f'Extracting CSV file(s) from {filename}.')
    from zipfile import ZipFile
    with ZipFile(filename, 'r') as zip_file:
        zip_file.extractall()

def load_data():
    try:
        if not os.path.isfile(XTrainCSVFilename) or not os.path.isfile(YTrainCSVFilename):
            download_and_unzip(TrainDataZipFilename)
    except DownloadError as e:
        raise DownloadError(
            f'Could not find CSV file {XTrainCSVFilename} or {YTrainCSVFilename} and downloading {e.filename} failed.\n{e.url}'
        ) from e.__cause__

    try:
        if not os.path.isfile(XTestCSVFilename):
            download_and_unzip(TestDataZipFilename)
    except DownloadError as e:
        raise DownloadError(
            f'Could not find CSV file {XTestCSVFilename} and downloading {e.filename} failed.\n{e.url}'
        ) from e.__cause__

    print('Reading CSV files')
    XTrain = load_signals_from_csv(XTrainCSVFilename)
    XTest  = load_signals_from_csv(XTestCSVFilename)
    YTrainInt = pd.read_csv(YTrainCSVFilename, dtype=int, header=None).squeeze("columns")
    return XTrain, YTrainInt, XTest

#%% main
if __name__ == "__main__":
    np.random.seed(21)

    XTrain, YTrainInt, XTest = load_data()

    print("\nClass counts:\n", pd.Series(YTrainInt).value_counts())
    print("==> no balancing required.\n")

    # Preprocessing: speichert XTest_preprocessed.npy in data/
    Xtr, Ytr, Xval, Yval = preprocess(
        XTrain=XTrain,
        YTrainInt=YTrainInt,
        XTest=XTest,
        n_classes=4,
        val_ratio=0.15,
        seed=21,
        save_dir="data",
    )

    # Model bauen
    input_shape = (Xtr.shape[1], Xtr.shape[2])  # (timesteps, channels)
    model = build_cnn_1d(input_shape=input_shape, n_classes=4, lr=0.001)
    model.summary()

    # Trainieren + beste Weights speichern
    best_weight_path = "models/best.weights.h5"
    train_model(model, Xtr, Ytr, Xval, Yval, weight_path=best_weight_path)

    print(f"\nFertig. Beste Weights: {best_weight_path}")
    print("Preprocessed Test liegt hier: data/XTest_preprocessed.npy")