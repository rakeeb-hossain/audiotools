import librosa
import numpy as np


def append_filename(path: str, suffix: str):
    # What is the extension of the file?
    parts = path.split('.')
    extension = parts[-1]
    return f"{parts[0]}{suffix}.{extension}"

def load_audio_and_resample(path: str, sr: int = 44100):
    audio, sr = librosa.load(path, mono=False, sr=sr)
    
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] > 2:
        if audio.shape[0] == 6:
            # Convert 6-channel audio to stereo by summing the channels
            left = audio[0] + 0.707*audio[2] + 0.707*audio[4]
            right = audio[1] + 0.707*audio[3] + 0.707*audio[5]
            audio = np.vstack((left, right))
        else:
            raise ValueError(f'Unsupported number of channels: {audio.shape[0]}')

    return audio
