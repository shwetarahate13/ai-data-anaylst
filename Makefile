.PHONY: setup data test demo eval eval-offline app api docker docker-compose clean

setup:
	python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

data:
	python3 -m analyst generate-data

test:
	python3 -m pytest -v

demo:
	python3 -m analyst investigate "Why did revenue drop in Q3?" --report --trace

eval:
	python3 -m eval.run_eval

eval-offline:
	python3 -m eval.run_eval --offline

app:
	streamlit run app/streamlit_app.py

api:
	uvicorn api.main:app --reload --port 8000

docker:
	docker build -t ai-data-analyst . && docker run --rm -p 8501:8501 --env-file .env ai-data-analyst

docker-compose:
	docker compose up --build

clean:
	rm -rf data/*.csv data/*.duckdb reports/*.md reports/why-*.html .pytest_cache
