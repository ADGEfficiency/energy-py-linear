./energypylinear.egg-info/PKG-INFO: setup.py
	pip install -r requirements.txt
	pip install -e .

test: ./energypylinear.egg-info/PKG-INFO
	pytest tests

####

expt-space-between: ./energypylinear.egg-info/PKG-INFO

# ./nem-data/setup.py: requirements
# 	rm -rf ./nem-data
# 	git clone git@github.com:ADGEfficiency/nem-data
# 	pip3 install nem-data/.
# 	pip3 install -q -r nem-data/requirements.txt

# #  TODO ref specific file
# ~/nem-data/trading-price/: ./nem-data/setup.py
# 	nem -s 2014-01 -e 2020-12 -r trading-price

# ~/nem-data/nemde/: ./nem-data/setup.py
# 	nem -s 2014-01 -e 2020-12 -r nemde
