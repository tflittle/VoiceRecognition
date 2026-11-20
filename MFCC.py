import numpy as np
import wave


SAMPLE_RATE = 16000
FRAME_LENGTH = 400 #(25ms)
HOP_LENGTH = 160 #(10ms)
FFT_SIZE = 512
MEL_FILTERS = 26
MFCC_COEFICCIENTS = 13

#################
### Functions ###
#################

def load_wav(filename):
    with wave.open(str(filename), "rb") as wav:
        sample_rate = wav.getframerate()
        num_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()

        raw = wav.readframes(wav.getnframes())

    if sample_rate != SAMPLE_RATE:
        raise ValueError("Wrong sample rate")

    if num_channels != 1:
        raise ValueError("Expected mono audio")

    if sample_width != 2:
        raise ValueError("Expected 16-bit PCM")

    audio = np.frombuffer(raw, dtype=np.int16)

    return audio.astype(np.float32)

def create_frames(audio):
    frames = []

    for start in range(0, len(audio) - FRAME_LENGTH + 1, HOP_LENGTH):
        frame = audio[start:start + FRAME_LENGTH]

        frames.append(frame)

    return np.array(frames)

def apply_window(frames):
    window = np.hamming(FRAME_LENGTH)

    return frames * window

def power_spectrum(frames):
    fft_result = np.fft.rfft(frames, n=FFT_SIZE)

    magnitude = np.abs(fft_result)

    power = (magnitude ** 2) / FFT_SIZE

    return power

def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (10 ** (mel / 2595) - 1)

def create_mel_filterbank():

    low_mel = hz_to_mel(0)
    high_mel = hz_to_mel(SAMPLE_RATE / 2)

    mel_points = np.linspace(low_mel, high_mel, MEL_FILTERS + 2)

    hz_points = mel_to_hz(mel_points)

    bins = np.floor((FFT_SIZE + 1) * hz_points / SAMPLE_RATE).astype(int)

    filterbank = np.zeros((MEL_FILTERS, FFT_SIZE // 2 + 1))

    for m in range(1, MEL_FILTERS + 1):

        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        for k in range(left, center):
            filterbank[m - 1, k] = (
                (k - left)
                / (center - left)
            )

        for k in range(center, right):
            filterbank[m - 1, k] = (
                (right - k)
                / (right - center)
            )

    return filterbank

def apply_mel_filterbank(power):
    filterbank = create_mel_filterbank()

    mel_energy = np.dot(
        power,
        filterbank.T
    )

    mel_energy = np.maximum(
        mel_energy,
        1e-10
    )

    log_mel = np.log(mel_energy)

    return log_mel

def dct(log_mel):
    num_frames = log_mel.shape[0]

    mfcc = np.zeros(
        (num_frames, MFCC_COEFICCIENTS)
    )

    for frame in range(num_frames):

        for coefficient in range(MFCC_COEFICCIENTS):

            total = 0

            for m in range(MEL_FILTERS):

                total += (
                    log_mel[frame, m]
                    * np.cos(
                        np.pi
                        * coefficient
                        * (m + 0.5)
                        / MEL_FILTERS
                    )
                )

            mfcc[frame, coefficient] = total

    return mfcc

def calculate_mfcc(filename):

    audio = load_wav(filename)

    audio = trim_silence(audio)

    audio = fix_audio_length(audio)   

    frames = create_frames(audio)

    frames = apply_window(frames)

    power = power_spectrum(frames)

    log_mel = apply_mel_filterbank(power)

    mfcc = dct(log_mel)

    return mfcc

def trim_silence(audio, threshold_ratio=0.08):

    peak = np.max(np.abs(audio))

    if peak == 0:
        return audio

    threshold = peak * threshold_ratio
    active = np.where(np.abs(audio) > threshold)[0]

    if len(active) == 0:
        return audio

    start = active[0]
    end = active[-1] + 1

    return audio[start:end]

def fix_audio_length(audio):
    target_length = SAMPLE_RATE  # 16000

    if len(audio) > target_length:
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]

    elif len(audio) < target_length:
        total_padding = target_length - len(audio)

        left_padding = total_padding // 2
        right_padding = total_padding - left_padding

        audio = np.pad(
            audio,
            (left_padding, right_padding)
        )

    return audio
