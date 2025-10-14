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
import itertools as it

import numpy as np

from pennylane.devices import QubitDevice
from pennylane.exceptions import DeviceError
from pennylane.ops import (
    BasisState,
    QubitUnitary,
    CRZ,
    PhaseShift,
    Adjoint,
    StatePrep,
)

import qulacs.gate as gate
from qulacs import QuantumCircuit, QuantumState, Observable

from ._version import __version__


GPU_SUPPORTED = True
try:
    from qulacs import QuantumStateGpu
except ImportError:
    GPU_SUPPORTED = False

phase_shift = lambda phi: np.array([[1, 0], [0, cmath.exp(1j * phi)]])

# Multi-qubit gates are represented in the convention of Qulacs
# E.g., for a controlled operation the first qubit is the target and the second
# qubit is the control with consecutive wires
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, cmath.exp(-1j * theta / 2), 0, 0],
        [0, 0, 1, 0],
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
    pennylane_requires = ">=0.43.0"
    version = __version__
    author = "Steven Oud and Xanadu"
    gpu_supported = GPU_SUPPORTED

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
        "returns_state": True,
    }

    _operation_map = {
        "StatePrep": None,
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
        "PhaseShift": phase_shift,
    }

    _observable_map = {
        "PauliX": "X",
        "PauliY": "Y",
        "PauliZ": "Z",
        "Identity": "I",
        "Hadamard": None,
        "Hermitian": None,
        "Prod": None,
    }

    operations = _operation_map.keys()
    observables = _observable_map.keys()

    # Add inverse gates to _operation_map
    _operation_map.update({k + ".inv": v for k, v in _operation_map.items()})

    def __init__(self, wires, shots=None, gpu=False, **kwargs):
        super().__init__(wires=wires, shots=shots)

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

        self._pre_rotated_state = self._state.copy()

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        self.apply_operations(operations)
        self._pre_rotated_state = self._state.copy()

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
            if i > 0 and isinstance(op, (BasisState, StatePrep)):
                raise DeviceError(
                    "Operation {} cannot be used after other Operations have already been applied "
                    "on a {} device.".format(op.name, self.short_name)
                )
            inverse = False
            if isinstance(op, Adjoint):
                inverse = True
                op = op.base

            if isinstance(op, StatePrep):
                self._apply_qubit_state_vector(op)
            elif isinstance(op, BasisState):
                self._apply_basis_state(op)
            elif isinstance(op, QubitUnitary):
                self._apply_qubit_unitary(op, inverse)
            elif isinstance(op, (CRZ, PhaseShift)):
                self._apply_matrix(op, inverse)
            else:
                self._apply_gate(op, inverse)

    def _expand_state(self, state_vector, wires):
        """Expands state vector to more wires"""
        basis_states = np.array(list(it.product([0, 1], repeat=len(wires))))

        # get basis states to alter on full set of qubits
        unravelled_indices = np.zeros((2 ** len(wires), self.num_wires), dtype=int)
        unravelled_indices[:, wires] = basis_states

        # get indices for which the state is changed to input state vector elements
        ravelled_indices = np.ravel_multi_index(unravelled_indices.T, [2] * self.num_wires)

        state = np.zeros([2**self.num_wires], dtype=np.complex128)
        state[ravelled_indices] = state_vector
        state_vector = state.reshape([2] * self.num_wires)

        return state_vector.flatten()

    def _apply_qubit_state_vector(self, op):
        """Initialize state with a state vector"""
        wires = self.map_wires(op.wires)
        input_state = op.parameters[0]

        if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        if len(wires) != self.num_wires or sorted(wires) != wires:
            input_state = self._expand_state(input_state, wires)
        input_state = _reverse_state(input_state)

        # call qulacs' state initialization
        self._state.load(input_state)

    def _apply_basis_state(self, op):
        """Initialize a basis state"""
        wires = op.wires
        par = op.parameters

        n_basis_state = len(par[0])

        if not set(par[0]).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")
        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # translate from PennyLane to Qulacs wire order
        bits = np.zeros(self.num_wires, dtype=int)
        bits[wires] = par[0]
        bits = bits[::-1]

        basis_state = 0
        for bit in bits:
            basis_state = (basis_state << 1) | bit

        # call qulacs' basis state initialization
        self._state.set_computational_basis(basis_state)

    def _apply_qubit_unitary(self, op, inverse=False):
        """Apply unitary to state"""
        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires(op.wires)
        par = op.parameters

        if len(par[0]) != 2 ** len(device_wires):
            raise ValueError("Unitary matrix must be of shape (2**wires, 2**wires).")

        if inverse:
            par[0] = par[0].conj().T

        # reverse wires (could also change par[0])
        reverse_wire_labels = device_wires.tolist()[::-1]
        unitary_gate = gate.DenseMatrix(reverse_wire_labels, par[0])
        self._circuit.add_gate(unitary_gate)
        unitary_gate.update_quantum_state(self._state)

    def _apply_matrix(self, op, inverse=False):
        """Apply predefined gate-matrix to state (must follow qulacs convention)"""
        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires(op.wires)
        par = op.parameters

        mapped_operation = self._operation_map[op.name]
        if inverse:
            mapped_operation = self._get_inverse_operation(mapped_operation, device_wires, par)

        if callable(mapped_operation):
            gate_matrix = mapped_operation(*par)
        else:
            gate_matrix = mapped_operation

        # gate_matrix is already in correct order => no wire-reversal needed
        dense_gate = gate.DenseMatrix(device_wires.labels, gate_matrix)
        self._circuit.add_gate(dense_gate)
        gate.DenseMatrix(device_wires.labels, gate_matrix).update_quantum_state(self._state)

    def _apply_gate(self, op, inverse=False):
        """Apply native qulacs gate"""

        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires(op.wires)
        par = op.parameters

        mapped_operation = self._operation_map[op.name]
        if inverse:
            mapped_operation = self._get_inverse_operation(mapped_operation, device_wires, par)

        # Negating the parameters such that it adheres to qulacs
        par = np.negative(par)

        # mapped_operation is already in correct order => no wire-reversal needed
        self._circuit.add_gate(mapped_operation(*device_wires.labels, *par))
        mapped_operation(*device_wires.labels, *par).update_quantum_state(self._state)

    @staticmethod
    def _get_inverse_operation(mapped_operation, device_wires, par):
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
                    mat = reduce(np.kron, [np.eye(2)] * len(device_wires)).astype(complex)
                    mat[-len(g) :, -len(g) :] = g

                    # mat follows PL convention => reverse wire-order
                    reverse_wire_labels = device_wires.tolist()[::-1]
                    gate_mat = gate.DenseMatrix(reverse_wire_labels, np.conj(mat).T)
                    return gate_mat

        return inverse_operation

    def analytic_probability(self, wires=None):
        """Return the (marginal) analytic probability of each computational basis state."""
        if self._state is None:
            return None

        all_probs = self._abs(self.state) ** 2
        prob = self.marginal_prob(all_probs, wires)
        return prob

    def expval(self, observable, **kwargs):
        if self.shots is None:
            qulacs_observable = Observable(self.num_wires)
            if isinstance(observable.name, list):
                observables = [self._observable_map[obs] for obs in observable.name]
            elif observable.name == "Prod":
                observables = [self._observable_map[obs.name] for obs in observable.operands]
            else:
                observables = [self._observable_map[observable.name]]

            if None not in observables:
                applied_wires = self.map_wires(observable.wires).tolist()
                opp = " ".join([f"{obs} {applied_wires[i]}" for i, obs in enumerate(observables)])

                qulacs_observable.add_operator(1.0, opp)
                return qulacs_observable.get_expectation_value(self._pre_rotated_state)

            # exact expectation value
            if callable(observable.eigvals):
                eigvals = self._asarray(observable.eigvals(), dtype=self.R_DTYPE)
            else:  # older version of pennylane
                eigvals = self._asarray(observable.eigvals, dtype=self.R_DTYPE)
            prob = self.probability(wires=observable.wires)
            return self._dot(eigvals, prob)

        # estimate the ev
        return np.mean(self.sample(observable))

    @property
    def state(self):
        # returns the state after all operations are applied
        return _reverse_state(self._state.get_vector())

    def reset(self):
        self._state.set_zero_state()
        self._pre_rotated_state = self._state.copy()
        self._circuit = QuantumCircuit(self.num_wires)
