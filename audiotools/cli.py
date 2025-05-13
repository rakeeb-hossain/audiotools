import os
import click
from pathlib import Path


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
@click.argument("output_path", type=click.Path())
def glue(audio_paths, output_path):
	if len(audio_paths) < 2:
		raise click.UsageError("At least two audio files are required to glue.")
	from audiotools.glue import glue_waveforms
	glue_waveforms(audio_paths, output_path)

@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.argument("name", type=str)
def train_svc(audio_path, name):
	from audiotools.svc import train_svc_model
	outpath = Path("output") / name
	os.makedirs(outpath, exist_ok=True)

	audio_path = Path(audio_path)
	train_svc_model(audio_path, outpath)

if __name__ == "__main__":
	cli()
	
