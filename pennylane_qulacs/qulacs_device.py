# Copyright 2020 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Base device class for PennyLane-Qulacs
======================================

**Module name:** :mod:`pennylane_qulacs.qulacs_device`

.. currentmodule:: pennylane_qulacs.qulacs_device

This Device implements all the :class:`~pennylane.device.QubitDevice` methods,
for using the Qulacs simulator as PennyLane device.

Classes
-------

.. autosummary::
   QulacsDevice

----
"""

import numpy as np

from scipy.linalg import block_diag

from pennylane import QubitDevice, DeviceError

import qulacs.gate as gate
from qulacs import QuantumCircuit, QuantumState

from . import __version__


GPU_SUPPORTED = True
try:
    from qulacs import QuantumStateGpu
except ImportError:
    GPU_SUPPORTED = False


I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

rx = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X
ry = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y
rz = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
    ]
)


def _reverse_state(state_vector):
    """Reverse the qubit order for a vector of amplitudes.
    Args:
        state_vector (iterable[complex]): vector containing the amplitudes
    Returns:
        list[complex]
    """
    state_vector = np.array(state_vector)
    N = int(np.log2(len(state_vector)))
    reversed_state = state_vector.reshape([2] * N).T.flatten()
    return list(reversed_state)


# tolerance for numerical errors
tolerance = 1e-10


class QulacsDevice(QubitDevice):
    """Qulacs device"""
    name = "Qulacs device"
    short_name = "qulacs.simulator"
    pennylane_requires = ">=0.5.0"
    version = __version__
    author = "Steven Oud and Xanadu"
    gpu_supported = GPU_SUPPORTED

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True
    }

    _operation_map = {
        "QubitStateVector": None,
        "BasisState": None,
        "QubitUnitary": None,
        "Toffoli": gate.TOFFOLI,
        "CSWAP": gate.FREDKIN,
        "CRZ": crz,
        "SWAP": gate.SWAP,
        "CNOT": gate.CNOT,
        "CZ": gate.CZ,
        "S": gate.S,
        "T": gate.T,
        "RX": gate.RX,
        "RY": gate.RY,
        "RZ": gate.RZ,
        "Rot": None,
        "PauliX": gate.X,
        "PauliY": gate.Y,
        "PauliZ": gate.Z,
        "Hadamard": gate.H,
        # TODO: Does the device have a phase shift?
       # "PhaseShift": gate.
    }

    _observable_map = {
        "PauliX": X,
        "PauliY": Y,
        "PauliZ": Z,
        "Hadamard": H,
        "Identity": I,
        #"Hermitian": hermitian
    }

    operations = _operation_map.keys()
    observables = _observable_map.keys()

    # Add inverse gates to _operation_map
    _operation_map.update({k + ".inv": v for k, v in _operation_map.items()})

    def __init__(self, wires, shots=1000, analytic=True, gpu=False, **kwargs):
        super().__init__(wires=wires, shots=shots, analytic=analytic)

        if gpu:
            if not QulacsDevice.gpu_supported:
                raise DeviceError(
                    "GPU not supported with installed version of qulacs. "
                    "Please install 'qulacs-gpu' to use GPU simulation."
                )

            self._state = QuantumStateGpu(self.num_wires)
        else:
            self._state = QuantumState(self.num_wires)

        self._circuit = QuantumCircuit(self.num_wires)

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        self.apply_operations(operations)

        # Rotating the state for measurement in the computational basis
        self.apply_operations(rotations)

    def apply_operations(self, operations):
        """Apply the circuit operations to the state.

        This method serves as an auxiliary method to :meth:`~.QulacsDevice.apply`.

        Args:
            operations (List[pennylane.Operation]): operations to be applied
        """

        for op in operations:
            wires = op.wires
            par = op.parameters

            mapped_op = self._operation_map[op.name]
            if op.name.endswith(".inv"):
                # if an inverse variant of the operation exists
                try:
                    mapped_operation = getattr(gate, mapped_op.get_name() + "dag")
                except AttributeError:
                    # if the operation is hard-coded
                    try:
                        mapped_operation = np.conj(mapped_op).T
                    except TypeError:
                        # else redefine the operation as the inverse matrix
                        mapped_operation = lambda *p: gate.DenseMatrix(
                            wires, np.conj(
                                mapped_op(*p).get_matrix()
                            ).T
                        )
            else:
                mapped_operation = mapped_op

            if op.name == "QubitStateVector":
                input_state = par[0]
                input_state = _reverse_state(input_state)

                if len(input_state) != 2**len(wires):
                    raise ValueError("State vector must be of length 2**wires.")
                if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
                    raise ValueError("Sum of amplitudes-squared does not equal one.")
                # call qulac"s state initialization
                self._state.load(input_state)

            elif op.name == "BasisState":
                # translate from PennyLane to Qulacs wire order
                bits = par[0][::-1]
                n_basis_state = len(bits)

                if not set(bits).issubset({0, 1}):
                    raise ValueError("BasisState parameter must consist of 0 or 1 integers.")
                if n_basis_state != len(wires):
                    raise ValueError("BasisState parameter and wires must be of equal length.")

                basis_state = 0
                for bit in bits:
                    basis_state = (basis_state << 1) | bit

                # call qulac's basis state initialization
                self._state.set_computational_basis(basis_state)

            elif op.name == "QubitUnitary":

                if len(par[0]) != 2 ** len(wires):
                    raise ValueError("Unitary matrix must be of shape (2**wires, 2**wires).")

                # either reverse wires (or change par[0]; harder)
                wires = wires[::-1]
                unitary_gate = gate.DenseMatrix(wires, par[0])
                self._circuit.add_gate(unitary_gate)
                unitary_gate.update_quantum_state(self._state)

            elif op.name == "Rot":
                # Negating the parameters such that it adheres to qulacs
                par = np.negative(op.parameters)

                if len(wires) != 1:
                    raise ValueError("Rotation gate can only be applied on a single wire.")

                self._circuit.add_gate(gate.RZ(wires[0], par[0]))
                gate.RZ(wires[0], par[0]).update_quantum_state(self._state)
                self._circuit.add_gate(gate.RY(wires[0], par[1]))
                gate.RY(wires[0], par[1]).update_quantum_state(self._state)
                self._circuit.add_gate(gate.RZ(wires[0], par[2]))
                gate.RZ(wires[0], par[2]).update_quantum_state(self._state)

            elif op.name == "CRZ":
                if callable(mapped_operation):
                    gate_matrix = mapped_operation(*par)
                else:
                    gate_matrix = mapped_operation

                # gate_matrix is already in correct order => no wire-reversal needed
                dense_gate = gate.DenseMatrix(wires, gate_matrix)
                self._circuit.add_gate(dense_gate)
                gate.DenseMatrix(wires, gate_matrix).update_quantum_state(self._state)

            else:
                # Negating the parameters such that it adheres to qulacs
                par = np.negative(op.parameters)

                # mapped_operation is already in correct order => no wire-reversal needed
                self._circuit.add_gate(mapped_operation(*wires, *par))
                mapped_operation(*wires, *par).update_quantum_state(self._state)

    def analytic_probability(self, wires=None):
        """Return the (marginal) analytic probability of each computational basis state."""
        if self._state is None:
            return None

        wires = wires or range(self.num_wires)

        all_probs = self._abs(self.state) ** 2
        prob = self.marginal_prob(all_probs, wires)
        return prob

    @property
    def state(self):
        return _reverse_state(self._state.get_vector())

    def reset(self):
        self._state.set_zero_state()
        self._circuit = QuantumCircuit(self.num_wires)
