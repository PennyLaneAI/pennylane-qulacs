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

# Swapping the order of the target and control qubits due to qulacs ordering
CSWAP = block_diag(SWAP, I, I)

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
# Swapping the order of the target and control qubits due to qulacs ordering
toffoli = np.diag([1 for i in range(8)])
toffoli[0:2, 0:2] = np.array([[0, 1], [1, 0]])


def hermitian(*args):
    r"""Input validation for an arbitary Hermitian expectation.
    Args:
        args (array): square hermitian matrix
    Returns:
        array: square hermitian matrix
    """
    A = np.asarray(args[0])

    if A.shape[0] != A.shape[1]:
        raise ValueError("Expectation must be a square matrix.")

    if not np.allclose(A, A.conj().T):
        raise ValueError("Expectation must be Hermitian.")

    return A


# tolerance for numerical errors
tolerance = 1e-10


class QulacsDevice(QubitDevice):
    """Qulacs device"""
    name = "Qulacs device"
    short_name = "qulacs.simulator"
    pennylane_requires = ">=0.5.0"
    version = __version__
    author = "Steven Oud and Xanadu"

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True
    }

    _operation_map = {
        "QubitStateVector": None,
        "BasisState": None,
        "QubitUnitary": None,
        # TODO: test Toffolis functioning
        #"Toffoli": toffoli,
        #"CSWAP": CSWAP,
        #"CRZ": crz,
        "SWAP": gate.SWAP,
        "CNOT": gate.CNOT,
        "CZ": gate.CZ,
        "S": gate.S,
        "S.inv": gate.Sdag,
        "T": gate.T,
        "T.inv": gate.Tdag,
        "RX": gate.RX,
        "RY": gate.RY,
        "RZ": gate.RZ,
        "Rot": None,
        "PauliX": gate.X,
        "PauliY": gate.Y,
        "PauliZ": gate.Z,
        "Hadamard": gate.H
#        "PhaseShift": gate.
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

    def __init__(self, wires, shots=1000, analytic=True, gpu=False, **kwargs):
        super().__init__(wires=wires, shots=shots, analytic=analytic)

        if gpu:
            if not GPU_SUPPORTED:
                raise DeviceError(
                    "GPU not supported with installed version of qulacs. "
                    "Please install 'qulacs-gpu' to use GPU simulation."
                )

            self._state = QuantumStateGpu(self.num_wires)
        else:
            self._state = QuantumState(self.num_wires)

        self._circuit = QuantumCircuit(self.num_wires)

    def apply(self, operations):

        for i, op in enumerate(operations):
            # revert the wire numbering such that it adheres to qulacs
            wires = op.wires.tolist()[::-1]
            par = op.parameters

            if i > 0 and op.name in {"BasisState", "QubitStateVector"}:
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(op, self.short_name)
                )

            if op.name == "QubitStateVector":
                input_state = par[0]

                if len(input_state) != 2**len(wires):
                    raise ValueError("State vector must be of length 2**wires.")
                if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
                    raise ValueError("Sum of amplitudes-squared does not equal one.")
                # call qulac"s state initialization
                self._state.load(par[0])

            elif op.name == "BasisState":

                # reorder
                bits = par[0][::-1]
                n_basis_state = len(bits)

                if not set(bits).issubset({0, 1}):
                    raise ValueError("BasisState parameter must consist of 0 or 1 integers.")
                if n_basis_state != len(wires):
                    raise ValueError("BasisState parameter and wires must be of equal length.")

                basis_state = 0
                for bit in bits:
                    basis_state = (basis_state << 1) | bit
                # call qulac"s basis state initialization
                self._state.set_computational_basis(basis_state)

            elif op.name == "QubitUnitary":
                if len(par[0]) != 2 ** len(wires):
                    raise ValueError("Unitary matrix must be of shape (2**wires, 2**wires).")

                unitary_gate = gate.DenseMatrix(wires, par[0])
                self._circuit.add_gate(unitary_gate)
                unitary_gate.update_quantum_state(self._state)

            elif op.name == "Rot":

                # Negating the parameters such that it adheres to qulacs
                par = np.negative(op.parameters)

                self._circuit.add_gate(gate.RZ(wires[0], par[0]))
                gate.RZ(wires[0], par[0]).update_quantum_state(self._state)
                self._circuit.add_gate(gate.RY(wires[0], par[1]))
                gate.RY(wires[0], par[1]).update_quantum_state(self._state)
                self._circuit.add_gate(gate.RZ(wires[0], par[2]))
                gate.RZ(wires[0], par[2]).update_quantum_state(self._state)

            elif op.name in ("CRZ", "Toffoli", "CSWAP"):
                mapped_operation = self._operation_map[op.name]
                if callable(mapped_operation):

                    gate_matrix = mapped_operation(*par)
                else:
                    # basis_states = np.array(list(itertools.product([0, 1], repeat=len(wires))))
                    # perm = np.ravel_multi_index(basis_states[:, np.argsort(np.argsort(wires))].T, [2] * len(wires))

                    gate_matrix = mapped_operation

                dense_gate = gate.DenseMatrix(wires, gate_matrix)
                self._circuit.add_gate(dense_gate)
                gate.DenseMatrix(wires, gate_matrix).update_quantum_state(self._state)

            else:
                # Negating the parameters such that it adheres to qulacs
                par = np.negative(op.parameters)

                mapped_operation = self._operation_map[op.name]
                self._circuit.add_gate(mapped_operation(*wires, *par))
                mapped_operation(*wires, *par).update_quantum_state(self._state)

    def analytic_probability(self, wires=None):

        if self._state is None:
            return None

        wires = wires or range(self.num_wires)
        # 0,1 means that the qubit is observed, and 2 means no measurement.
        measured_values = [1 if w in wires else 2 for w in range(self.num_wires)]
        prob = self._state.get_marginal_probability(measured_values=measured_values)

        return prob

    @property
    def state(self):
        return self._state.get_vector()

    def reset(self):
        self._state.set_zero_state()
        self._circuit = QuantumCircuit(self.num_wires)

