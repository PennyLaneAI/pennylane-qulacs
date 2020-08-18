import re

from setuptools import setup


with open('./pennylane_qulacs/__init__.py') as f:
    version, = re.findall('__version__ = \"(.*)\"', f.read())

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='pennylane_qulacs',
    description='PennyLane plugin for Qulacs',
    version=version,
    long_description=long_description,
    install_requires=[
        'pennylane>=0.11.0',
        'numpy',
        'scipy',
    ],
    extras_require={
        "cpu": ["qulacs>=0.1.10.1"],
        "gpu": ["qulacs-gpu>=0.1.10.1"],
    },
    packages=['pennylane_qulacs'],
    entry_points={
        'pennylane.plugins': [
            'qulacs.simulator = pennylane_qulacs.qulacs_device:QulacsDevice'
        ]
    }
)
