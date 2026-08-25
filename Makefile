.PHONY: start boot setup clean doctor api ui test check engine-check play play3d demo reflex ab player-ab player-reflex-ab

start boot:
	./racelab start

setup:
	./racelab setup

clean:
	./racelab clean

doctor:
	./racelab doctor

api:
	.venv/bin/uvicorn harness.api:app --app-dir backend --reload --env-file .env --port 8000

ui:
	npm run dev

test:
	./racelab test

check:
	npm run check

engine-check:
	./racelab engine-check

play:
	./racelab play2d

play3d:
	./racelab play3d

demo:
	.venv/bin/harness demo

reflex:
	PYTHONPATH=backend:scripts .venv/bin/python scripts/run_reflex_demo.py --both

ab:
	PYTHONPATH=backend:scripts .venv/bin/python scripts/run_harness_ab.py

player-ab:
	PYTHONPATH=backend:scripts .venv/bin/python scripts/run_player_ab.py

player-reflex-ab:
	PYTHONPATH=backend:scripts .venv/bin/python scripts/run_player_reflex_ab.py
