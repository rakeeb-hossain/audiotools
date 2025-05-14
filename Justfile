default:

run *args:
	poetry run python audiotools/cli.py {{args}}

svc *args:
	poetry run svc {{args}}

tb *args:
	poetry run tensorboard {{args}}
