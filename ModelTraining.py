import numpy as np
import wave
import pandas as pd
import MFCC as mfcc

from pathlib import Path
from MFCC import calculate_mfcc
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

### Variables ###

rng = 37

### Dataset Handling ###

DATASET_FOLDER = Path(str(r"C:\Users\trent\VoiceRecognition\TrainingData"))

CLASSES = ["Yes","No"]

def load_dataset():
    X = []
    y = []

    for class_index, class_name in enumerate(CLASSES):

        folder = DATASET_FOLDER / class_name

        print("Loading:", class_name)

        for wav_file in folder.glob("*.wav"):

            mfcc = calculate_mfcc(wav_file)

            #Convert MFFC matrix to vector
            features = mfcc.flatten()

            X.append(features)
            y.append(class_index)

    X = np.array(X, dtype = np.float32)
    y = np.array(y, dtype = np.int32)

    return X, y

X, y = load_dataset()

print("X shape:", X.shape)
print("y shape:", y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = rng, stratify = y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes = (32,),activation = "relu", solver = "adam", max_iter = 1000, random_state = rng)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("Accuracy:", accuracy)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

np.savez("trained_model.npz", scaler_mean = scaler.mean_, scaler_scale = scaler.scale_, W1 = model.coefs_[0], B1 = model.intercepts_[0], W2 = model.coefs_[1], B2 = model.intercepts_[1], classes = np.array(CLASSES))