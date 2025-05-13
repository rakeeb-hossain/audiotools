default:

run:
	poetry run python audiotools/cli.py

svc *args:
	poetry run svc {{args}}
