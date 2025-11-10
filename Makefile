run:
	python app.py

install:
	pip install -r requirements.txt

clean:
	rm -rf __pycache__

.PHONY: run install clean
