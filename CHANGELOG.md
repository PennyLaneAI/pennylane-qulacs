# Release 0.16.0-dev

### New features

### Improvements

### Bug fixes

Fixes an issue when the wrong results are returned when passing non-consecutive wires to `QubitStateVector`.
[(#25)](https://github.com/PennyLaneAI/pennylane-qulacs/pull/25)

### Breaking changes

### Documentation

### Contributors

This release contains contributions from (in alphabetical order):

Theodor Isacsson

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
