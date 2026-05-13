import os
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, X_train, Y_train, X_val, Y_val, weight_path: str):
    os.makedirs(os.path.dirname(weight_path), exist_ok=True)

    es = EarlyStopping(
        monitor="val_loss",
        patience=50,
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    cp = ModelCheckpoint(
        weight_path,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="min",
        save_freq="epoch",
    )

    rlrop = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=10,
        verbose=1,
        min_lr=1e-5,
    )

    history = model.fit(
        X_train, Y_train,
        validation_data=(X_val, Y_val),
        epochs=200,
        batch_size=32,
        callbacks=[es, cp, rlrop],
        verbose=1
    )
    return history