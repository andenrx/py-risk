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
    ],
    extras_require={
        "nn": [
            "torch",
            "torch-scatter",
            "torch-sparse",
            "torch-geometric",
        ]
    },
    dependency_links=[
        "https://github.com/ImparaAI/monte-carlo-tree-search/archive/refs/tags/v1.3.1.tar.gz",
        "https://data.pyg.org/whl/torch-1.9.0+cu111.html",
        "https://data.pyg.org/whl/torch-1.9.0+cu111.html",
    ],
)
