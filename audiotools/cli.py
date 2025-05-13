import click
from audiotools.stem_splitter import split_stems_mdx23

@click.group
def cli():
	pass

@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
def split_stems(audio_path):
	print(audio_path)
	split_stems_mdx23(audio_path)

if __name__ == "__main__":
	cli()
	
