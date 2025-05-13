import numpy as np
import os
from pathlib import Path
import shutil

from audiotools.utils import load_audio_and_resample
from so_vits_svc_fork.preprocessing.preprocess_resample import preprocess_resample
from so_vits_svc_fork.preprocessing.preprocess_split import preprocess_split
from so_vits_svc_fork.preprocessing.preprocess_flist_config import preprocess_config
from so_vits_svc_fork.preprocessing.preprocess_hubert_f0 import preprocess_hubert_f0
from so_vits_svc_fork.train import train as train_so_vits

from rich.console import Console

c = Console()


"""
`train_svc_model` assumes all audio files are cleaned, have one speaker, and 
have no background music.
"""
def train_svc_model(
    source_path: Path, 
    output_dir: Path,
    skip_preprocessing: bool = False,
):
    config_dir = output_dir / 'config'
    config_path = config_dir / 'config.json'
    if not skip_preprocessing:
        # Step 1: collect raw dataset
        c.print('Step 1: Collecting raw dataset...', style='bold yellow')
        dataset_dir = output_dir / 'dataset'
        os.makedirs(dataset_dir, exist_ok=True)

        # Is source_path a file or a directory?
        if source_path.is_file():
            # Create a directory for the speaker
            shutil.copy(source_path, dataset_dir / source_path.name)
        else:
            # Load all audio files in the directory
            audio_files = list(source_path.rglob('*.wav|*.mp3|*.m4a|*.flac'))
            for file in audio_files:
                shutil.copy(file, dataset_dir / file.name)

        # Step 2: split
        c.print('Step 2: Splitting dataset...', style='bold yellow')
        split_dir = output_dir / 'split'
        os.makedirs(split_dir, exist_ok=True)
        preprocess_split(dataset_dir, split_dir, 44100)

        # Step 3: resample
        c.print('Step 3: Resampling dataset...', style='bold yellow')
        resampled_dir = output_dir / 'resampled'
        os.makedirs(resampled_dir, exist_ok=True)
        preprocess_resample(split_dir, resampled_dir, 44100)

        # Step 4: config
        c.print('Step 4: Generating config...', style='bold yellow')
        os.makedirs(config_dir, exist_ok=True)
        preprocess_config(
            resampled_dir,
            config_dir / 'train.txt',
            config_dir / 'val.txt',
            config_dir / 'test.txt',
            config_path,
            'so-vits-svc-4.0v1',
        )

        # Step 5: extract f0
        c.print('Step 5: Extracting f0...', style='bold yellow')
        preprocess_hubert_f0(resampled_dir, config_path, f0_method='crepe')

    # Step 6: train
    c.print('Step 6: Training...', style='bold yellow')
    model_dir = output_dir / 'model'
    os.makedirs(model_dir, exist_ok=True)
    train_so_vits(config_path, model_dir)
