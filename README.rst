PennyLane-Qulacs Plugin
#######################


.. image:: https://img.shields.io/github/workflow/status/PennyLaneAI/pennylane-qulacs/Tests/master?logo=github&style=flat-square
    :alt: GitHub Workflow Status (branch)
    :target: https://github.com/PennyLaneAI/pennylane-qulacs/actions?query=workflow%3ATests

.. image:: https://img.shields.io/codecov/c/github/PennyLaneAI/pennylane-qulacs/master.svg?logo=codecov&style=flat-square
    :alt: Codecov coverage
    :target: https://codecov.io/gh/PennyLaneAI/pennylane-qulacs

.. image:: https://img.shields.io/codefactor/grade/github/PennyLaneAI/pennylane-qulacs/master?logo=codefactor&style=flat-square
    :alt: CodeFactor Grade
    :target: https://www.codefactor.io/repository/github/pennylaneai/pennylane-qulacs

.. image:: https://img.shields.io/readthedocs/pennylane-qulacs.svg?logo=read-the-docs&style=flat-square
    :alt: Read the Docs
    :target: https://pennylane-qulacs.readthedocs.io

.. image:: https://img.shields.io/pypi/v/PennyLane-qulacs.svg?style=flat-square
    :alt: PyPI
    :target: https://pypi.org/project/PennyLane-qulacs

.. image:: https://img.shields.io/pypi/pyversions/PennyLane-qulacs.svg?style=flat-square
    :alt: PyPI - Python Version
    :target: https://pypi.org/project/PennyLane-qulacs

.. header-start-inclusion-marker-do-not-remove

The PennyLane-Qulacs plugin integrates the Qulacs quantum computing framework with PennyLane's
quantum machine learning capabilities.

`PennyLane <https://pennylane.readthedocs.io>`__ is a cross-platform Python library for quantum machine
learning, automatic differentiation, and optimization of hybrid quantum-classical computations.

`Qulacs <https://github.com/qulacs>`__ is a software library for quantum computing, written in C++ and with GPU support.

.. header-end-inclusion-marker-do-not-remove

The plugin documentation can be found here: `<https://pennylane-qulacs.readthedocs.io/en/latest/>`__.

Features
========

* Provides access to Qulacs' simulator backend via the ``qulacs.simulator`` device

* Support for all PennyLane core functionality

.. benchmarks-start-inclusion-marker-do-not-remove

Benchmarks
==========

We ran a 100 executions of 4 layer quantum neural
network `strongly entangling layer <https://pennylane.readthedocs.io/en/latest/code/api/pennylane.templates.layers.StronglyEntanglingLayers.html>`_
and compared the runtimes between CPU and GPU.

.. image:: https://raw.githubusercontent.com/soudy/pennylane-qulacs/master/images/qnn_cpu_vs_gpu.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

|

.. image:: https://raw.githubusercontent.com/soudy/pennylane-qulacs/master/images/qulacs_table.png
    :align: center
    :width: 60%
    :target: javascript:void(0);

|


.. benchmarks-end-inclusion-marker-do-not-remove


.. installation-start-inclusion-marker-do-not-remove

Installation
============

This plugin requires Python version 3.6 or above, as well as PennyLane and
Qulacs. Installation of this plugin, as well as all dependencies, can be done
using ``pip``:

.. code-block:: bash

    $ pip install pennylane-qulacs["cpu"]

Note that you need to include whether to install the CPU version
(``pennylane-qulacs["cpu"]``) or the GPU version (``pennylane-qulacs["gpu"]``)
of Qulacs for it to be installed correctly. Otherwise Qulacs will need to be
installed independently:

.. code-block:: bash

    pip install qulacs pennylane-qulacs

Alternatively, you can install PennyLane-Qulacs from the `source code
<https://github.com/PennyLaneAI/pennylane-qulacs>`__ by navigating to the top
directory and running:

.. code-block:: bash

    $ python setup.py install

.. note::

    Qulacs supports parallelized executions via OpenMP. To set the number of
    threads to use during simulations you need to update the environment
    variable ``OMP_NUM_THREADS``. It can be set using the UNIX command:

    ``export OMP_NUM_THREADS = 8``

    where 8 can be replaced by the number of threads that you wish to use. By
    default Qulacs uses all available threads. To restore the default behaviour,
    simply remove the environment variable. It can be done using the UNIX command:

    ``unset OMP_NUM_THREADS``

    See the `OpenMP documentation page for OMP_NUM_THREADS
    <https://www.openmp.org/spec-html/5.0/openmpse50.html>`__ or `here
    <https://en.wikipedia.org/wiki/Environment_variable>`__ for more details on
    how to use environment variables.

Dependencies
~~~~~~~~~~~~

PennyLane-Qulacs requires the following libraries be installed:

* `Python <http://python.org/>`__ >= 3.6

as well as the following Python packages:

* `PennyLane <http://pennylane.readthedocs.io/>`__ >= 0.9
* `Qulacs <https://docs.qulacs.org/en/latest/>`__  >= 0.1.9


If you currently do not have Python 3 installed, we recommend
`Anaconda for Python 3 <https://www.anaconda.com/download/>`__, a distributed version of Python packaged
for scientific computation.


Tests
~~~~~

To test that the PennyLane-Qulacs plugin is working correctly you can run

.. code-block:: bash

    $ make test

in the source folder.

Documentation
~~~~~~~~~~~~~

To build the HTML documentation, go to the top-level directory and run:

.. code-block:: bash

  $ make docs


The documentation can then be found in the ``doc/_build/html/`` directory.

.. installation-end-inclusion-marker-do-not-remove

Contributing
============

We welcome contributions - simply fork the repository of this plugin, and then make a
`pull request <https://help.github.com/articles/about-pull-requests/>`__ containing your contribution.
All contributers to this plugin will be listed as authors on the releases.

We also encourage bug reports, suggestions for new features and enhancements, and even links to cool projects
or applications built on PennyLane.

Authors
=======

PennyLane-Qulacs is the work of `many contributors <https://github.com/PennyLaneAI/pennylane-qulacs/graphs/contributors>`__.

If you are doing research using PennyLane and PennyLane-Qulacs, please cite `our paper <https://arxiv.org/abs/1811.04968>`__:

    Ville Bergholm, Josh Izaac, Maria Schuld, Christian Gogolin, M. Sohaib Alam, Shahnawaz Ahmed,
    Juan Miguel Arrazola, Carsten Blank, Alain Delgado, Soran Jahangiri, Keri McKiernan, Johannes Jakob Meyer,
    Zeyue Niu, Antal Száva, and Nathan Killoran.
    *PennyLane: Automatic differentiation of hybrid quantum-classical computations.* 2018. arXiv:1811.04968

.. support-start-inclusion-marker-do-not-remove

Support
=======

- **Source Code:** https://github.com/PennyLaneAI/pennylane-qulacs
- **Issue Tracker:** https://github.com/PennyLaneAI/pennylane-qulacs/issues
- **PennyLane Forum:** https://discuss.pennylane.ai

If you are having issues, please let us know by posting the issue on our Github issue tracker, or
by asking a question in the forum.

.. support-end-inclusion-marker-do-not-remove
.. license-start-inclusion-marker-do-not-remove

License
=======

The PennyLane-Qulacs plugin is **free** and **open source**, released under
the `Apache License, Version 2.0 <https://www.apache.org/licenses/LICENSE-2.0>`__.

.. license-end-inclusion-marker-do-not-remove
