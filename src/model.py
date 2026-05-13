from keras.models import Sequential
from keras.layers import Input, Conv1D, BatchNormalization, ReLU, MaxPooling1D, Dropout, Flatten, Dense
from keras.optimizers import Adam

def build_cnn_1d(input_shape, n_classes=4, lr=0.001):
    model = Sequential()

    model.add(Input(shape=input_shape))  # (timesteps, channels=3)

    model.add(Conv1D(filters=16, kernel_size=7, padding="same", use_bias=False, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))


    model.add(Conv1D(filters=8, kernel_size=5, padding="same", use_bias=False, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=8, kernel_size=3, padding="same", use_bias=False, activation="relu"))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(64, activation="relu"))

    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )
    return model