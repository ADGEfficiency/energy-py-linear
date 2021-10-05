./energypylinear.egg-info/PKG-INFO: setup.py
	pip install -r requirements.txt
	pip install -e .

test: ./energypylinear.egg-info/PKG-INFO
	pytest tests

####

expt-space-between: ./energypylinear.egg-info/PKG-INFO ./nem-data/setup.py ~/nem-data/data/TRADINGPRICE/2020-12/clean.parquet ~/nem-data/data/nemde/2020-12-31/clean.parquet
	echo ""

./nem-data/setup.py:
	rm -rf ./nem-data
	git clone git@github.com:ADGEfficiency/nem-data
	pip3 install nem-data/.
	pip3 install -q -r nem-data/requirements.txt

~/nem-data/data/TRADINGPRICE/2020-12/clean.parquet: ./nem-data/setup.py
	nem -s 2014-01 -e 2020-12 -r trading-price

~/nem-data/data/nemde/2020-12-31/clean.parquet: ./nem-data/setup.py
	nem -s 2014-01 -e 2020-12 -r nemde
