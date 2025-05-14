from pathlib import Path
import soundfile as sf
import numpy as np

from audiotools.utils import load_audio_and_resample

"""
glue_waveforms layers the waveforms of audio files together. 
It pads the shorter file with zeros to match the length of the longer file.
"""
def glue_waveforms(audio_files: list[Path], output_path: Path):
    # Load the audio files
    audio_files = [load_audio_and_resample(file)[0] for file in audio_files]

    # Pad the length of all files to the length of the longest file
    max_length = max(len(file) for file in audio_files)
    audio_files = [np.pad(file, (0, max_length - len(file)), mode='constant') for file in audio_files]

    # Concatenate the audio files
    glued = np.sum(audio_files, axis=0)

    # Normalize audio
    glue_max, glue_min = glued.max(), glued.min()
    if glue_max > 1.0 or glue_min < -1.0:
        glued = glued / max(abs(glue_max), abs(glue_min))

    # Save the glued audio (NOTE: soundfile expects the transpose)
    sf.write(output_path, glued.T, 44100, format='WAV', subtype='FLOAT')
