from setuptools import setup, find_packages

setup(
    name='py-risk',
    version='0.1',
    description='Python Risk',
    url='https://github.com/andenrx/py-risk',
    author='Andrew Bauer',
    author_email='abauer7@asu.edu',
    packages=['risk'],
    install_requires=[
       "numpy",
       "requests",
       "python-igraph",
       "imparaai-montecarlo",
       "wonderwords",
       "pyyaml",
       "pygad @ https://github.com/andenrx/GeneticAlgorithmPython/archive/master.zip",
       "scipy", # only for objectives.py
    ],
    extras_require={
        "nn": [
            "torch",
            "torch-scatter",
            "torch-sparse",
            "torch-geometric",
        ]
    },
)
