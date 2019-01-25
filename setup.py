from setuptools import setup, find_packages


setup(
    name='energypylinear',

    version='0.0.3',
    description='linear programming for energy systems',
    author='Adam Green',
    author_email='adam.green@adgefficiency.com',
    url='http://www.adgefficiency.com/',

    packages=find_packages(exclude=['tests', 'tests.*']),

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=['PuLP']
)
