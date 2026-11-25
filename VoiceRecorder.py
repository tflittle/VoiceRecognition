import numpy as np
import serial
import soundfile as sf
import time
import os

from MFCC import calculate_mfcc


MODEL_FILE = "trained_model.npz"
AUDIO_FILE = "new_recording.wav"

OUTPUT_FOLDER = r"/home/trent/VoiceRecognition/VoiceReference/disable"


os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ser = serial.Serial('/dev/ttyACM1', 921600, timeout = 5)

SAMPLE_RATE = 16000
NUM_SAMPLES = 16000
NUM_BYTES = NUM_SAMPLES * 2

for samples in range(251,301):

    filename = os.path.join(OUTPUT_FOLDER, f"disable_{samples}.wav")
    input("\nPress Enter, then say the word...")

    time.sleep(0.2)

    ser.write(b'r')

    raw = ser.read(NUM_BYTES)

    print("\nBytes received:", len(raw))

    audio = np.frombuffer(raw, dtype=np.int16)
    print("Samples:", len(audio))
    print("Minimum:", np.min(audio))
    print("Maximum:", np.max(audio))
    
    print("RMS:", np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

    audio = np.frombuffer(raw, dtype = np.int16)

    sf.write(filename, audio, SAMPLE_RATE, subtype = "PCM_16")

    print(f"\nSaved recording {samples}")

ser.close()

# -------------------------------------------------
# Load exported model data
# -------------------------------------------------

# model_data = np.load(
#     MODEL_FILE,
#     allow_pickle=True
# )

# mean = model_data["scaler_mean"]
# scale = model_data["scaler_scale"]

# W1 = model_data["W1"]
# B1 = model_data["B1"]

# W2 = model_data["W2"]
# B2 = model_data["B2"]

# classes = model_data["classes"]


# # -------------------------------------------------
# # Activation functions
# # -------------------------------------------------

# def relu(x):
#     return np.maximum(0, x)


# def sigmoid(x):
#     return 1.0 / (1.0 + np.exp(-x))


# # -------------------------------------------------
# # Load audio and calculate MFCC
# # -------------------------------------------------

# mfcc = calculate_mfcc(AUDIO_FILE)

# print("MFCC shape:", mfcc.shape)


# # -------------------------------------------------
# # Flatten MFCC
# #
# # 98 x 13 -> 1274
# # -------------------------------------------------

# input_data = mfcc.flatten()

# print("Input size:", len(input_data))


# # -------------------------------------------------
# # Normalize exactly like training
# # -------------------------------------------------

# input_data = (
#     input_data - mean
# ) / scale


# # -------------------------------------------------
# # Hidden layer
# #
# # input: 1274
# # W1:    1274 x 32
# # output: 32
# # -------------------------------------------------

# hidden = np.dot(
#     input_data,
#     W1
# ) + B1

# hidden = relu(hidden)


# # -------------------------------------------------
# # Output layer
# # -------------------------------------------------

# output = np.dot(
#     hidden,
#     W2
# ) + B2

# probability = sigmoid(output)

# probability = float(
#     probability.flatten()[0]
# )


# # -------------------------------------------------
# # Prediction
# # -------------------------------------------------

# print()
# print("Probability:", probability)

# # Assuming:
# #
# # classes[0] = YES
# # classes[1] = NO

# if probability >= 0.5:
#     prediction = classes[1]
# else:
#     prediction = classes[0]

# print("Prediction:", prediction)