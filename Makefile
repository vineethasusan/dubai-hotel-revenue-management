export PYTHONPATH := src

.PHONY: install pipeline dashboard test lint docker

install:    ## install dependencies
	pip install -r requirements-dev.txt
pipeline:   ## simulate market, train models, compare pricing policies -> outputs/report.md
	python -m hotelrm.pipeline
dashboard:  ## http://localhost:8501
	streamlit run dashboard.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker build -t dubai-hotel-revenue-management . && docker run -p 8501:8501 dubai-hotel-revenue-management
