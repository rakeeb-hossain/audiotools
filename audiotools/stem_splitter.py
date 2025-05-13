import librosa
import numpy as np
import soundfile as sf
from vendor.ZFTurbo_MDX23.inference import EnsembleDemucsMDXMusicSeparationModel
from rich.console import Console

c = Console()

def split_stems_mdx23(audio_path):
    # Load the audio file
    audio, sr = librosa.load(audio_path, mono=False, sr=44100)

    # Handle different channel configurations and convert to stereo
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio], axis=0)
    elif audio.shape[0] > 2:
        if audio.shape[0] == 6:
            # Convert 6-channel audio to stereo by summing the channels
            left = audio[0] + 0.707*audio[2] + 0.707*audio[4]
            right = audio[1] + 0.707*audio[3] + 0.707*audio[5]
            audio = np.vstack((left, right))
        else:
            raise ValueError(f"Unsupported number of channels: {audio.shape[0]}")
    c.print(f"Loaded audio with shape: {audio.shape}, Sample rate: {sr}", style="bold yellow")

    # Setup model options
    options = {
        "overlap_small": 0.5,
        "overlap_large": 0.6,
        "chunk_size": 1500000,
        "large_gpu": True, # Use GPU optimization
    }
    model = EnsembleDemucsMDXMusicSeparationModel(options)
    
    # Separate the stems
    result, sample_rates = model.separate_music_file(
        audio.T,
        sr,
        only_vocals=True,
        total_files=1,
    )
    
    print(result.keys())
    