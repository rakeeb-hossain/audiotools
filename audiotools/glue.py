import soundfile as sf
import numpy as np

from audiotools.utils import load_audio_and_resample

"""
glue_waveforms layers the waveforms of audio files together. 
It pads the shorter file with zeros to match the length of the longer file.
"""
def glue_waveforms(audio_files: list[str], output_path: str):
    # Load the audio files
    audio_files = [load_audio_and_resample(file)[0] for file in audio_files]

    # Pad the length of all files to the length of the longest file
    max_length = max(len(file) for file in audio_files)
    audio_files = [np.pad(file, (0, max_length - len(file)), mode='constant') for file in audio_files]

    # Concatenate the audio files
    glued = np.sum(audio_files, axis=0)

    # Save the glued audio
    sf.write(output_path, glued, 44100)
