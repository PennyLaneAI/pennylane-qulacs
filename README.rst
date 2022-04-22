PennyLane-Qrack Plugin
#######################

The PennyLane-Qrack plugin integrates the Qrack quantum computing framework with PennyLane's quantum machine learning capabilities.

This plugin is addapted from the `PennyLane-Qulacs plugin, <https://github.com/PennyLaneAI/pennylane-qulacs>`__ under the Apache License 2.0, with many thanks to the original developers!

`PennyLane <https://pennylane.readthedocs.io>`__ is a cross-platform Python library for quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`vm6502q/qrack <https://github.com/vm6502q>`__ is a software library for quantum computing, written in C++ and with GPU support.

Features
========

* Provides access to a PyQrack simulator backend via the ``qrack.simulator`` device

Installation
============

This plugin requires Python version 3.6 or above, as well as PennyLane. Installation of this plugin, as well as all dependencies, can be done using ``pip``:

.. code-block:: bash

    $ pip install pennylane-qrack

Dependencies
~~~~~~~~~~~~

PennyLane-Qrack requires the following libraries be installed:

* `Python <http://python.org/>`__ >= 3.6

as well as the following Python packages:

* `PennyLane <http://pennylane.readthedocs.io/>`__ >= 0.9
* `PyQrack <https://github.com/vm6502q/pyqrack>`__  >= 0.12.1


If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`__, a distributed version of Python packaged
for scientific computation.


Tests
~~~~~

To test that the PennyLane-Qrack plugin is working correctly you can run

.. code-block:: bash

    $ make test

in the source folder.

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`__ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

Authors
=======

PennyLane-Qrack has been directly adapted by Daniel Strano from PennyLane-Qulacs. PennyLane-Qulacs is the work of `many contributors <https://github.com/PennyLaneAI/pennylane-qulacs/graphs/contributors>`__.

If you are doing research using PennyLane and PennyLane-Qulacs, please cite `their paper <https://arxiv.org/abs/1811.04968>`__:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

Support
=======

- **Source Code:** https://github.com/vm6502q/pennylane-qrack
- **Issue Tracker:** https://github.com/vm6502q/pennylane-qrack/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

License
=======

The PennyLane-Qrack plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__.
