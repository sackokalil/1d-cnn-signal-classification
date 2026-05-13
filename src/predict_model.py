import numpy as np

def predict_classes(model, X):
    prob = model.predict(X, verbose=0)
    return np.argmax(prob, axis=1).astype(int)