# PennyLane Qulacs Plugin
[PennyLane](https://pennylane.readthedocs.io) is a cross-platform Python
library for quantum machine learning, automatic differentiation, and
optimization of hybrid quantum-classical computations.

[Qulacs](http://qulacs.org/) is a high-performance quantum circuit
simulator for simulating large, noisy or parametric quantum circuits.
Implemented in C/C++ and with python interface, Qulacs achieved both high
speed circuit simulation and high usability.

This PennyLane plugin allows the use of the Qulacs simulator as device for PennyLane.

## Features
* Provides `qulacs.simulator` device to be used with PennyLane.
* Combine Qulacs high performance simulator with PennyLane's automatic differentiation and optimization.

## Installation
Installing the latest master version can be done directly using pip:
```
pip install git+https://github.com/soudy/pennylane-qulacs@master
```
or by cloning this repo:
```
git clone https://github.com/soudy/pennylane-qulacs
cd pennylane-qulacs
pip install .
```

# Benchmarks
We ran a 50 executions of 8 quantum neural network [strongly entangling layer](https://pennylane.readthedocs.io/en/latest/code/api/pennylane.templates.layers.StronglyEntanglingLayer.html) and compared the runtimes between CPU and GPU.

<p align="center">
  <img alt="Qulacs PennyLane plugin benchmarks" src="https://raw.githubusercontent.com/soudy/pennylane-qulacs/master/images/qnn_cpu_vs_gpu.png">
</p>
