\# 1D CNN Signal Classification for Time Series Data



A deep learning project for multivariate time-series classification using a 1D Convolutional Neural Network (1D CNN) implemented with TensorFlow/Keras.



The project classifies 3-channel signal data into 4 different classes.



The complete workflow includes:



\- automatic dataset downloading

\- preprocessing of multichannel signals

\- channel-wise standardization

\- 1D CNN training

\- model checkpointing

\- prediction generation

\- Kaggle-style CSV export



\---



\# Project Overview



This project uses:



\- TensorFlow / Keras

\- 1D Convolutional Neural Networks

\- Time-Series Classification

\- Signal Processing

\- EarlyStopping

\- ModelCheckpoint

\- ReduceLROnPlateau



The model is trained on multichannel sequential signal data.



\---



\# Features



\- Automatic dataset download and extraction

\- Multichannel signal preprocessing

\- Channel-wise standardization

\- 1D CNN architecture

\- Validation split

\- Automatic checkpoint saving

\- Prediction export for Kaggle submission

\- CSV-based signal loading



\---



\# Dataset



The project uses:



\- `TrainDataSignale.zip`

\- `XTestSignale.zip`



The datasets contain:



\- multivariate signal sequences

\- 3 channels per sample

\- 4 target classes



The datasets are automatically downloaded and extracted if not already available.



\---



\# Project Structure



```text

.

├── data/

│   ├── XTest\_preprocessed.npy

│   ├── mu.npy

│   └── sigma.npy

│

├── models/

│   └── best.weights.h5

│

├── output/

│   └── yPredict\_SackoKalil.csv

│

├── src/

│   ├── model.py

│   ├── predict\_model.py

│   ├── preprocessing.py

│   └── train\_model.py

│

├── do\_prediction.py

├── main.py

├── environment.yml

└── README.md

```



\---



\# Signal Preprocessing



The preprocessing pipeline includes:



\- loading multichannel CSV signals

\- reshaping signals into `(samples, timesteps, channels)`

\- shuffling

\- one-hot encoding

\- train/validation split

\- per-channel standardization



The preprocessing statistics:



\- `mu.npy`

\- `sigma.npy`



are automatically saved for reuse.



Implemented in:



```text

src/preprocessing.py

```



\---



\# 1D CNN Architecture



The classifier is based on a 1D Convolutional Neural Network.



\## Architecture



\- Conv1D (16 filters, kernel size 7)

\- MaxPooling1D



\- Conv1D (8 filters, kernel size 5)

\- MaxPooling1D



\- Conv1D (8 filters, kernel size 3)

\- MaxPooling1D



\- Flatten

\- Dense Layer (64 neurons)

\- Softmax Output Layer



The model uses:



\- ReLU activation

\- Adam optimizer

\- Categorical Crossentropy loss



\---



\# Training



The training pipeline uses:



\- EarlyStopping

\- ModelCheckpoint

\- ReduceLROnPlateau



Best weights are automatically saved inside:



```text

models/

```



Training is implemented in:



```text

src/train\_model.py

```



\---



\# Installation



\## 1. Clone the repository



```bash

git clone https://github.com/sackokalil/1d-cnn-signal-classification.git

cd 1d-cnn-signal-classification

```



\---



\## 2. Create the environment



Using Conda:



```bash

conda env create -f environment.yml

conda activate challenge2

```



\---



\# Run the Project



\## Train the model



```bash

python main.py

```



This will:



\- download the datasets

\- preprocess the signals

\- train the 1D CNN

\- save the best weights

\- save preprocessing statistics



\---



\## Generate predictions



```bash

python do\_prediction.py

```



This will:



\- load the preprocessed test data

\- reconstruct the CNN architecture

\- load saved weights

\- generate predictions

\- export predictions as CSV



\---



\# Output



Prediction results are saved as:



```text

output/yPredict\_SackoKalil.csv

```



\## CSV Format



```csv

id,category

1,2

2,0

3,1

...

```



This format is compatible with Kaggle submissions.



\---



\# Technologies Used



\- Python

\- TensorFlow / Keras

\- NumPy

\- Pandas



\---



\# Author



Kalil Sacko



Master Student in Computer Science  

University of Applied Sciences Bochum(Hochschule Bochum)



\---



\# Notes



\- Signals are standardized independently per channel.

\- The preprocessing statistics are computed only on training data.

\- Predictions are generated using Softmax probabilities and `argmax`.

\- The project automatically downloads the required datasets if missing.

