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
Base device class for PennyLane-Qulacs.
"""
from functools import reduce
import math, cmath

import numpy as np
from scipy.linalg import block_diag

from pennylane import QubitDevice, DeviceError
from pennylane.ops import QubitStateVector, BasisState, QubitUnitary, CRZ, PhaseShift

import qulacs.gate as gate
from qulacs import QuantumCircuit, QuantumState

from . import __version__


GPU_SUPPORTED = True
try:
    from qulacs import QuantumStateGpu
except ImportError:
    GPU_SUPPORTED = False

phase_shift = lambda phi: np.array(
    [
        [1, 0],
        [0, cmath.exp(1j * phi)]
     ]
)
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, cmath.exp(-1j * theta / 2), 0],
        [0, 0, 0, cmath.exp(1j * theta / 2)],
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
    N = int(math.log2(len(state_vector)))
    reversed_state = state_vector.reshape([2] * N).T.flatten()
    return reversed_state


# tolerance for numerical errors
tolerance = 1e-10


class QulacsDevice(QubitDevice):
    """Qulacs device"""
    name = "Qulacs device"
    short_name = "qulacs.simulator"
    pennylane_requires = ">=0.10.0"
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
        "PauliX": gate.X,
        "PauliY": gate.Y,
        "PauliZ": gate.Z,
        "Hadamard": gate.H,
        "PhaseShift": phase_shift
    }

    operations = _operation_map.keys()
    observables = {"PauliX", "PauliY", "PauliZ", "Identity", "Hadamard", "Hermitian"}

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

        self._pre_rotated_state = self._state

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        self.apply_operations(operations)
        self._pre_rotated_state = self._state

        # Rotating the state for measurement in the computational basis
        if rotations:
            self.apply_operations(rotations)

    def apply_operations(self, operations):
        """Apply the circuit operations to the state.

        This method serves as an auxiliary method to :meth:`~.QulacsDevice.apply`.

        Args:
            operations (List[pennylane.Operation]): operations to be applied
        """

        for i, op in enumerate(operations):
            if i > 0 and isinstance(op, (QubitStateVector, BasisState)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(op.name, self.short_name)
                )

            if isinstance(op, QubitStateVector):
                self._apply_qubit_state_vector(op)
            elif isinstance(op, BasisState):
                self._apply_basis_state(op)
            elif isinstance(op, QubitUnitary):
                self._apply_qubit_unitary(op)
            elif isinstance(op, (CRZ, PhaseShift)):
                self._apply_matrix(op)
            else:
                self._apply_gate(op)

    def _apply_qubit_state_vector(self, op):
        """Initialize state with a state vector"""
        wires = op.wires
        input_state = op.parameters[0]

        if len(input_state) != 2**len(wires):
            raise ValueError("State vector must be of length 2**wires.")
        if input_state.ndim != 1 or len(input_state) != 2 ** len(wires):
            raise ValueError("State vector must be of length 2**wires.")
        if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        input_state = _reverse_state(input_state)

        # call qulacs' state initialization
        self._state.load(input_state)

    def _apply_basis_state(self, op):
        """Initialize a basis state"""
        wires = op.wires
        par = op.parameters

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

        # call qulacs' basis state initialization
        self._state.set_computational_basis(basis_state)

    def _apply_qubit_unitary(self, op):
        """Apply unitary to state"""
        wires = op.wires
        par = op.parameters

        if len(par[0]) != 2 ** len(wires):
            raise ValueError("Unitary matrix must be of shape (2**wires, 2**wires).")

        if op.inverse:
            par[0] = par[0].conj().T

        # reverse wires (could also change par[0])
        unitary_gate = gate.DenseMatrix(wires[::-1], par[0])
        self._circuit.add_gate(unitary_gate)
        unitary_gate.update_quantum_state(self._state)

    def _apply_matrix(self, op):
        """Apply predefined gate-matrix to state (must follow qulacs convention)"""
        wires = op.wires
        par = op.parameters

        mapped_operation = self._operation_map[op.name]
        if op.inverse:
            mapped_operation = self._get_inverse_operation(mapped_operation, wires, par)

        if callable(mapped_operation):
            gate_matrix = mapped_operation(*par)
        else:
            gate_matrix = mapped_operation

        # gate_matrix is already in correct order => no wire-reversal needed
        dense_gate = gate.DenseMatrix(wires, gate_matrix)
        self._circuit.add_gate(dense_gate)
        gate.DenseMatrix(wires, gate_matrix).update_quantum_state(self._state)

    def _apply_gate(self, op):
        """Apply native qulacs gate"""
        wires = op.wires
        par = op.parameters

        mapped_operation = self._operation_map[op.name]
        if op.inverse:
            mapped_operation = self._get_inverse_operation(mapped_operation, wires, par)

        # Negating the parameters such that it adheres to qulacs
        par = np.negative(par)

        # mapped_operation is already in correct order => no wire-reversal needed
        self._circuit.add_gate(mapped_operation(*wires, *par))
        mapped_operation(*wires, *par).update_quantum_state(self._state)

    @staticmethod
    def _get_inverse_operation(mapped_operation, wires, par):
        """Return the inverse of an operation"""
        if mapped_operation is None:
            return mapped_operation

        # if an inverse variant of the operation exists
        try:
            inverse_operation = getattr(gate, mapped_operation.get_name() + "dag")
        except AttributeError:
            # if the operation is hard-coded
            try:
                if callable(mapped_operation):
                    inverse_operation = np.conj(mapped_operation(*par)).T
                else:
                    inverse_operation = np.conj(mapped_operation).T
            # if mapped_operation is a qulacs.gate and np.conj is applied on it
            except TypeError:
                # else, redefine the operation as the inverse matrix
                def inverse_operation(*p):
                    # embed the gate in a unitary matrix with shape (2**wires, 2**wires)
                    g = mapped_operation(*p).get_matrix()
                    mat = reduce(np.kron, [np.eye(2)]*len(wires)).astype(complex)
                    mat[-len(g):, -len(g):] = g

                    # mat follows PL convention => reverse wire-order
                    gate_mat = gate.DenseMatrix(wires[::-1], np.conj(mat).T)
                    return gate_mat

        return inverse_operation

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
        # returns the state after all operations are applied
        return _reverse_state(self._pre_rotated_state.get_vector())

    def reset(self):
        self._state.set_zero_state()
        self._pre_rotated_state = self._state
        self._circuit = QuantumCircuit(self.num_wires)
