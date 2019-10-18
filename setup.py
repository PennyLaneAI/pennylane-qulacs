import re

from setuptools import setup


with open('./pennylane_qulacs/__init__.py') as f:
    version, = re.findall('__version__ = \'(.*)\'', f.read())

with open('README.md', 'r') as f:
    long_description = f.read()

setup(
    name='PennyLane-Qulacs',
    description='PennyLane plugin for Qulacs',
    version=version,
    long_description=long_description,
    requirements=[
        'pennylane>=0.5.0',
        'numpy',
        'qulacs'
    ],
    entry_points={
        'pennylane.plugins': [
            'qulacs.simulator = pennylane_qulacs.qulacs_device:QulacsDevice'
        ]
    }
)
