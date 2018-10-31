try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup, find_packages

import tensorD

version = tensorD.__version__


def readme():
    with open('README.rst') as f:
        return f.read()


config = {
    'name': 'tensorD',
    'packages': find_packages(exclude=['doc']),
    'description': 'Tensor Decomposition via TensorFlow',
    'long_description': readme(),
    'author': 'Jinmian Ye, Liyang Hao, Siqi Liang',
    'author_email': 'jinmian.y@gmail.com',
    'version': version,
    'url': 'https://github.com/Large-Scale-Tensor-Decomposition/tensorD',
    'download_url': 'https://github.com/Large-Scale-Tensor-Decomposition/tensorD' + version,  # TODO : release version
    'install_requires': ['numpy', 'tensorflow>=1.0'],
    'license': 'MIT',
    'scripts': [],
    'classifiers': [
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3'
    ],
}

setup(**config)
