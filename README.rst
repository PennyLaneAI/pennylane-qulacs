PennyLane-Qrack Plugin
#######################

The PennyLane-Qrack plugin integrates the Qrack quantum computing framework with PennyLane's quantum machine learning capabilities.

This plugin is addapted from the `PennyLane-Qulacs plugin, <https://github.com/PennyLaneAI/pennylane-qulacs>`__ under the Apache License 2.0, with many thanks to the original developers!

`PennyLane <https://pennylane.readthedocs.io>`__ is a cross-platform Python library for quantum machine learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`unitaryfund/qrack <https://github.com/unitaryfund/qrack>`__ (formerly **vm6502q/qrack**) is a software library for quantum computing, written in C++ and with GPU support.

Features
========

* Provides access to a PyQrack simulator backend via the ``qrack.simulator`` device
* Provides access to a (C++) Qrack simulator backend for Catalyst (also) via the ``qrack.simulator`` device

Installation
============

This plugin requires Python version 3.6 or above, as well as PennyLane and the Qrack library.

You can choose to go the `releases <https://github.com/unitaryfund/qrack/releases>`__ page of Qrack to download a packaged artifact for your system and **install it in your system directories** (like `/usr` or `/usr/local` on Linux and UNIX based systems, including Mac), **or** you can opt to **build and install from source**, which might be easier, and this gives you maximum control over build configurations, like choice of CUDA over OpenCL GPU acceleration!

See the Qrack README and documentation for the many build options of qrack, but, after checking out the Qrack repository and entering its root folder, this might be the best and simplest way to build and install Qrack:

.. code-block:: bash

    $ mkdir _build
    $ cd _build
    $ cmake -DCPP_STD=14 ..
    $ make all -j$(nproc --all)
    $ sudo make install

After installing Qrack, installation of this plugin as well as all its Python dependencies can be done using ``pip`` (or ``pip3``, as appropriate):

.. code-block:: bash

    $ pip3 install pennylane-qrack

Dependencies
~~~~~~~~~~~~

PennyLane-Qrack requires the following libraries be installed:

* `Python <http://python.org/>`__ >= 3.8
* `Qrack <https://github.com/unitaryfund/qrack>`__ >= 9.0

as well as the following Python packages:

* `PennyLane <http://pennylane.readthedocs.io/>`__ >= 0.32
* `Catalyst <https://docs.pennylane.ai/projects/catalyst/en/stable/index.html>`__ >= 0.6
* `PyQrack <https://github.com/vm6502q/pyqrack>`__  >= 1.28.0


If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`__, a distributed version of Python packaged
for scientific computation.


Tests
~~~~~

To test that the PennyLane-Qrack plugin is working correctly you can run

.. code-block:: bash

    $ make test

in the source folder.

Documentation
=============
The API-doc can be generated locally by running
.. code-block:: bash

    $ make docs

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
