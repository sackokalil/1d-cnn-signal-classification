import os
import numpy as np
import pandas as pd

from src.model import build_cnn_1d
from src.predict_model import predict_classes

if __name__ == "__main__":
    # 1) Lade preprocessed Test
    XTest = np.load("data/XTest_preprocessed.npy")

    # 2) Rebuild Model (gleiche Architektur wie im Training!)
    input_shape = (XTest.shape[1], XTest.shape[2])
    model = build_cnn_1d(input_shape=input_shape, n_classes=4, lr=0.001)

    # 3) Lade beste Gewichte 
    weight_path = "models/best.weights.h5"
    model.load_weights(weight_path)

    # 4) Predict
    y_pred = predict_classes(model, XTest)

    # 5) CSV im Kaggle-Format: id,category 
    out_dir = "output"
    os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame({
        "id": np.arange(1, len(y_pred) + 1),
        "category": y_pred.astype(int)
    })


    out_path = os.path.join(out_dir, "yPredict_SackoKalil.csv")
    df.to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
    print(df.head(10))