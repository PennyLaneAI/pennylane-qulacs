from setuptools import setup


setup(
    requirements=[
        'pennylane>=0.5.0',
        'numpy'
    ],
    entry_points={'pennylane.plugins': [
        'default.qulacs = pennylane_qulacs.qulacs_device:QulacsDevice'
    ]}
)
