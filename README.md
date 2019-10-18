# PennyLane Qulacs Plugin
[PennyLane](https://pennylane.readthedocs.io) is a cross-platform Python
library for quantum machine learning, automatic differentiation, and
optimization of hybrid quantum-classical computations.

[Qulacs](http://qulacs.org/) Qulacs is high-performance quantum circuit
simulator for simulating large, noisy or parametric quantum circuits.
Implemented in C/C++ and with python interface, Qulacs achieved both high
speed circuit simulation and high usability.

This PennyLane plugin allows to use the simulator of Qulacs as devices for PennyLane.

## Features

* Provides `qulacs.simulator` device to be used with PennyLane.
* Combine Qulacs high performance simulator with PennyLane's automatic differentiation and optimization.
