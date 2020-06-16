The Simulator device
====================

You can instantiate the qulacs device in PennyLane as follows:

.. code-block:: python

    import pennylane as qml

    dev = qml.device('qulacs.simulator', wires=2)

This device can then be used just like other devices for the definition and evaluation of QNodes within PennyLane.
A simple quantum function that returns the expectation value of a measurement and depends on three classical input
parameters would look like:

.. code-block:: python

    @qml.qnode(dev)
    def circuit(x, y, z):
        qml.RZ(z, wires=[0])
        qml.RY(y, wires=[0])
        qml.RX(x, wires=[0])
        qml.CNOT(wires=[0, 1])
        return qml.expval(qml.PauliZ(wires=1))

You can then execute the circuit like any other function to get the quantum mechanical expectation value.

.. code-block:: python

    circuit(0.2, 0.1, 0.3)

Device options
~~~~~~~~~~~~~~

To run the qulacs device simulations on a GPU, set the custom ``qpu`` argument to ``True`` when creating the device.

.. code-block:: python

    dev = qml.device('qulacs.simulator', wires=2, gpu=True)

.. note::

    For GPU support, you need to have the ``qulacs-gpu`` version installed. Check the
    `Qulacs documentation <http://docs.qulacs.org/en/latest/intro/1_install.html>`_  for details.

Supported operations
~~~~~~~~~~~~~~~~~~~~

The ``qulacs.simulator`` device supports all PennyLane
`operations and observables <https://pennylane.readthedocs.io/en/stable/introduction/operations.html>`_, except from:

