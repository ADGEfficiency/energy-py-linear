./energypylinear.egg-info/PKG-INFO: setup.py
	pip install -r requirements.txt
	pip install -e .

test: ./energypylinear.egg-info/PKG-INFO
	pytest tests
