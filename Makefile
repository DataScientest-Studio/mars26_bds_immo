.PHONY: setup download-data clean

setup: download-data
	pip install -r requirements.txt

download-data:
	python download_data.py

clean:
	rm -rf data/
