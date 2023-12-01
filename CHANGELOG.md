# Release 0.33.0-dev

### New features since last release

### Improvements 🛠

### Breaking changes 💔

### Deprecations 👋

### Documentation 📝

### Bug fixes 🐛

### Contributors ✍️

This release contains contributions from (in alphabetical order):

---
# Release 0.32.0

### Breaking changes

* Support for Python 3.8 has been removed, and support for 3.11 has been added.
  [(#55)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/55)

### Improvements

* Added support for `qml.StatePrep` as a state preparation operation. 
  [(#54)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/54)

### Contributors

This release contains contributions from (in alphabetical order):

Mudit Pandey,
Jay Soni

---
# Release 0.29.0

### Improvements

* Removed support for in-place inversion of operations (e.g. `qml.PauliX(0).inv()`). Users should
  use `qml.adjoint` instead.
  [(#46)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/46)

### Contributors

This release contains contributions from (in alphabetical order):

Albert Mitjans Coma

---

# Release 0.28.0

### Breaking changes

* Removes testing for Python 3.7.
  [(#43)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/43)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee

---

# Release 0.24.0

### Improvements

* Adds the compatibility tag for Python 3.10 and removed it for 3.6.
  [(#31)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/31)

### Bug fixes

* Defines the missing `returns_state` entry of the `capabilities` dictionary of
  the device.
  [(#36)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/36)

* Updates the plugin to be compatible with the use of `Operator.eigvals` as a method.
  [(#35)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/35)

### Contributors

This release contains contributions from (in alphabetical order):

Christina Lee, Antal Száva

---

# Release 0.16.0

### Improvements

* Adds the compatibility tag for Python 3.9.
  [(#28)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/28)
  
### Bug fixes

* Fixed an issue where the wrong results are returned when passing non-consecutive wires to `QubitStateVector`.
  [(#25)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/25)

* Fixed issue where a state cannot be loaded into a Qulacs circuit when using a subset of wires with `BasisState`.
  [(#26)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/26)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Romain Moyard

# Release 0.15.0

### Breaking changes

* Removed the `analytic` argument to reflect the [shots
  refactor](https://github.com/PennyLaneAI/pennylane/pull/1079) in PennyLane. Analytic expectation
  values can still be computed by setting `shots=None`.
  [(#21)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/21)

### Contributors

This release contains contributions from (in alphabetical order):

Maria Schuld

# Release 0.14.0

## Bug fixes

* Adjusted the matrix for the `CRZ` operation used internally.
  [(#18)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/18)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Antal Száva

# Release 0.12.0

### Improvements

* Speeds up the computation of expectation values by using native Qulacs methods.
  [(#12)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/12)

### Documentation

* Updates the installation instructions, to indicate how to install the PennyLane-Qulacs
  plugin with either the CPU or GPU version of Qulacs.
  [(#14)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/14)

* Adds note to the documentation showing how to use multiple OpenMP threads by setting the
  `OMP_NUM_THREADS` environment variable.
  [(#14)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/14)

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson, Maria Schuld.

---

# Release 0.11.0

Initial public release.

### Contributors

This release contains contributions from:

Theodor Isacsson, Steven Oud, Maria Schuld, Antal Száva
