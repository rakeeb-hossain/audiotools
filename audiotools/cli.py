import os
import click
from pathlib import Path
from rich.console import Console

c = Console()

@click.group
def cli():
	pass

@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
def split_stems(audio_path):
	from audiotools.stem_splitter import split_stems_mdx23
	print(audio_path)
	split_stems_mdx23(audio_path)

@cli.command()
@click.argument("audio_paths", type=click.Path(exists=True), nargs=-1)
@click.option("-o", "--output", type=click.Path())
def glue(audio_paths, output):
	if len(audio_paths) < 2:
		raise click.UsageError("At least two audio files are required to glue.")
	audio_paths = [Path(path) for path in audio_paths]
	output = Path(output)

	c.print(f"Glueing {audio_paths} to {output}...", style="bold yellow")
	from audiotools.glue import glue_waveforms
	glue_waveforms(audio_paths, output)

@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("-n", "--name", required=True, type=str)
@click.option("--skip-preprocessing", default=False, is_flag=True)
def train_svc(audio_path, name, skip_preprocessing):
	from audiotools.svc import train_svc_model
	outpath = Path("output") / name
	os.makedirs(outpath, exist_ok=True)

	audio_path = Path(audio_path)
	train_svc_model(audio_path, outpath, skip_preprocessing)

@cli.command()
@click.argument("name", type=str)
@click.option("-i", "--infile", required=True, type=click.Path(exists=True))
@click.option("-o", "--outfile", required=True, type=click.Path())
@click.option("--separate", default=False, is_flag=True, help="Separate the vocals from input before inference and re-glue after.")
def infer_svc(name, infile, outfile, separate):
	from audiotools.svc import infer_svc_model
	import tempfile

	train_dir = Path("output") / name

	if not separate:
		infer_svc_model(train_dir, infile, outfile)
	else:
		from audiotools.stem_splitter import split_stems_mdx23
		from audiotools.glue import glue_waveforms
		import shutil

		with tempfile.TemporaryDirectory() as temp_dir:
			temp_dir = Path(temp_dir)
			incopy = temp_dir / "input.wav"
			shutil.copy(infile, incopy)

			c.print(f"Splitting stems from {infile}...", style="bold yellow")
			vocals_path, instrumental_path = split_stems_mdx23(incopy)

			c.print(f"Inferred SVC model for {vocals_path}...", style="bold yellow")
			svc_vocals_path = temp_dir / "svc_vocals.wav"
			infer_svc_model(train_dir, vocals_path, svc_vocals_path)

			c.print(f"Glueing {instrumental_path} and {svc_vocals_path}...", style="bold yellow")
			glue_waveforms([instrumental_path, svc_vocals_path], outfile)


if __name__ == "__main__":
	cli()
	
