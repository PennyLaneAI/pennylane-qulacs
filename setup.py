import re

from setuptools import setup


with open('./pennylane_qulacs/__init__.py') as f:
    version, = re.findall('__version__ = \'(.*)\'', f.read())

with open('README.rst', 'r') as f:
    long_description = f.read()

setup(
    name='pennylane_qulacs',
    description='PennyLane plugin for Qulacs',
    version=version,
    long_description=long_description,
    install_requires=[
        'pennylane>=0.10.0',
        'numpy',
        'scipy',
        'qulacs>=0.1.10.1'
    ],
    extras_require=[
        'qulacs-gpu',
    ],
    packages=['pennylane_qulacs'],
    entry_points={
        'pennylane.plugins': [
            'qulacs.simulator = pennylane_qulacs.qulacs_device:QulacsDevice'
        ]
    }
)
