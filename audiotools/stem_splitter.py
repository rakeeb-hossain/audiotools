import soundfile as sf
from audiotools.utils import append_filename, load_audio_and_resample
from vendor.ZFTurbo_MDX23.inference import EnsembleDemucsMDXMusicSeparationModel
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn, TimeElapsedColumn

c = Console()


class MDX23ProgressHandler:
    def __init__(self, console):
        self.console = console
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        )
        self.task = None

    def start(self):
        self.progress.start()
        self.task = self.progress.add_task("[bold blue]Separating stems...", total=100)

    def update(self, progress):
        if self.task is not None:
            self.progress.update(self.task, completed=progress)

    def finish(self):
        self.progress.stop()


def split_stems_mdx23(audio_path):
    # Load the audio file
    audio, sr = load_audio_and_resample(audio_path)
    c.print(f"Loaded audio with shape: {audio.shape}, Sample rate: {sr}", style="bold yellow")

    # Setup model options
    options = {
        "overlap_small": 0.5,
        "overlap_large": 0.6,
        "chunk_size": 1500000,
        "large_gpu": True, # Use GPU optimization
    }
    model = EnsembleDemucsMDXMusicSeparationModel(options)

    progress_handler = MDX23ProgressHandler(c)
    progress_handler.start()

    # Separate the stems
    result, sample_rates = model.separate_music_file(
        audio.T,
        sr,
        update_percent_func=progress_handler.update,
        only_vocals=True,
        total_files=1,
    )
    progress_handler.update(100)
    progress_handler.finish()

    if 'vocals' not in result:
        raise ValueError("No vocals found in the audio...")

    c.print("Extracting instrumental...", style="bold yellow")
    # Extract the instrumental by subtracting the vocals from the original audio
    instrumental = audio.T - result['vocals']

    # Save the vocals
    vocals_path = append_filename(audio_path, '_vocals')
    c.print(f"Saving vocals to {vocals_path}...", style="bold yellow")
    sf.write(vocals_path, result['vocals'], sr)

    # Save the instrumental
    instrumental_path = append_filename(audio_path, '_instrumental')
    c.print(f"Saving instrumental to {instrumental_path}...", style="bold yellow")
    sf.write(instrumental_path, instrumental, sr)
    