from setuptools import setup


setup(
    name='energypylinear',

    version='0.0.1',
    description='linear programming for energy systems',
    author='Adam Green',
    author_email='adam.green@adgefficiency.com',
    url='http://www.adgefficiency.com/',

    packages=['energypylinear/battery'],

    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    install_requires=['PuLP']
)
