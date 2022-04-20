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
Base device class for PennyLane-Qrack.
"""
from functools import reduce
import math, cmath
import itertools as it

import numpy as np

from pennylane import QubitDevice, DeviceError
from pennylane.ops import QubitStateVector, BasisState, QubitUnitary, CRZ, PhaseShift
from pennylane.wires import Wires

from pyqrack import QrackSimulator, Pauli

from . import __version__

# tolerance for numerical errors
tolerance = 1e-10


class QrackDevice(QubitDevice):
    """Qrack device"""

    name = "Qrack device"
    short_name = "qrack.simulator"
    pennylane_requires = ">=0.11.0"
    version = __version__
    author = "Daniel Strano, adapted from Steven Oud and Xanadu"

    _capabilities = {"model": "qubit", "tensor_observables": True, "inverse_operations": True}

    _observable_map = {
        "PauliX": Pauli.PauliX,
        "PauliY": Pauli.PauliY,
        "PauliZ": Pauli.PauliZ,
        "Identity": Pauli.PauliI,
        "Hadamard": None,
        "Hermitian": None,
    }

    observables = _observable_map.keys()
    operations = {
        "Toffoli",
        "CSWAP",
        "CRZ",
        "SWAP",
        "CNOT",
        "CZ",
        "S",
        "T",
        "RX",
        "RY",
        "RZ",
        "PauliX",
        "PauliY",
        "PauliZ",
        "Hadamard",
        "PhaseShift"
    }

    def __init__(self, wires, shots=None, **kwargs):
        super().__init__(wires=wires, shots=shots)

        self._state = QrackSimulator(self.num_wires)

        self._pre_rotated_state = QrackSimulator(cloneSid = self._state.sid)

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        self.apply_operations(operations)

        # Dealloc copy before creating a second copy.
        del self._pre_rotated_state

        self._pre_rotated_state = QrackSimulator(cloneSid = self._state.sid)

        # Rotating the state for measurement in the computational basis
        if rotations:
            self.apply_operations(rotations)

    def apply_operations(self, operations):
        """Apply the circuit operations to the state.

        This method serves as an auxiliary method to :meth:`~.QrackDevice.apply`.

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
            else:
                self._apply_gate(op)

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

        if len(input_state) != 2 ** len(wires):
            raise ValueError("State vector must be of length 2**wires.")
        if input_state.ndim != 1 or len(input_state) != 2 ** len(wires):
            raise ValueError("State vector must be of length 2**wires.")
        if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        if len(wires) != self.num_wires or sorted(wires) != wires:
            input_state = self._expand_state(input_state, wires)

        # call qrack' state initialization
        self._state.in_ket(input_state)

    def _apply_basis_state(self, op):
        """Initialize a basis state"""
        wires = op.wires
        par = op.parameters

        n_basis_state = len(par[0])

        if not set(par[0]).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")
        if n_basis_state != len(wires):
            raise ValueError("BasisState parameter and wires must be of equal length.")

        # translate from PennyLane to Qrack wire order
        bits = np.zeros(self.num_wires, dtype=int)
        bits[wires] = par[0]
        bits = bits[::-1]

        basis_state = 0
        for bit in bits:
            basis_state = (basis_state << 1) | bit

        for i in range(len(wires.labels)):
            if ((basis_state >> i) & 1) != self._state.m(wires.labels[i]):
                self._state.x(wires.labels[i])

    def _apply_qubit_unitary(self, op):
        """Apply unitary to state"""
        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires(op.wires)
        par = op.parameters

        if len(par[0]) != 2 ** len(device_wires):
            raise ValueError("Unitary matrix must be of shape (2**wires, 2**wires).")

        if op.inverse:
            par[0] = par[0].conj().T

        # reverse wires (could also change par[0])
        reverse_wire_labels = device_wires.tolist()[::-1]
        unitary_gate = gate.DenseMatrix(reverse_wire_labels, par[0])
        unitary_gate.update_quantum_state(self._state)

    def _apply_gate(self, op):
        """Apply native qrack gate"""

        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires(op.wires)
        par = op.parameters

        if op.name == "Toffoli" or op.name == "CNOT":
            self._state.mcx(device_wires.labels[1:], device_wires.labels[0])
        elif op.name == "CSWAP":
            self._state.mcswap(device_wires.labels[2:], device_wires.labels[0], device_wires.labels[1])
        elif op.name == "CRZ":
            if op.inverse:
                par[0] = -par[0]
            self._state.mcr(Pauli.PauliZ, math.pi * par[0], [device_wires.labels[1:]], device_wires.labels[0])
        elif op.name == "SWAP":
            self._state.swap(device_wires.labels[0], device_wires.labels[1])
        elif op.name == "CZ":
            self._state.mcz(device_wires.labels[1:], device_wires.labels[0])
        elif op.name == "S":
            if op.inverse:
                self._state.adjs(device_wires.labels[0])
            else:
                self._state.s(device_wires.labels[0])
        elif op.name == "T":
            if op.inverse:
                self._state.adjt(device_wires.labels[0])
            else:
                self._state.t(device_wires.labels[0])
        elif op.name == "RX":
            if op.inverse:
                par[0] = -par[0]
            self._state.r(Pauli.PauliX, par[0], device_wires.labels[0])
        elif op.name == "RY":
            if op.inverse:
                par[0] = -par[0]
            self._state.r(Pauli.PauliY, par[0], device_wires.labels[0])
        elif op.name == "RZ":
            if op.inverse:
                par[0] = -par[0]
            self._state.r(Pauli.PauliZ, par[0], device_wires.labels[0])
        elif op.name == "X":
            self._state.x(device_wires.labels[0])
        elif op.name == "Y":
            self._state.y(device_wires.labels[0])
        elif op.name == "Z":
            self._state.z(device_wires.labels[0])
        elif op.name == "H":
            self._state.h(device_wires.labels[0])
        elif op.name == "PhaseShift":
            if op.inverse:
                par[0] = -par[0]
            self._state.mtrx([1, 0, 0, cmath.exp(1j * par[0])], device_wires.labels[0])

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
            # if mapped_operation is a qrack.gate and np.conj is applied on it
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
            if isinstance(observable.name, list):
                observables = [self._observable_map[obs] for obs in observable.name]
            else:
                observables = [self._observable_map[observable.name]]

            if None not in observables:
                applied_wires = self.map_wires(observable.wires).tolist()
                opp = " ".join([f"{obs} {applied_wires[i]}" for i, obs in enumerate(observables)])

                return self._pre_rotated_state.measure_pauli(observables, list(range(self.num_wires)))

            # exact expectation value
            eigvals = self._asarray(observable.eigvals, dtype=self.R_DTYPE)
            prob = self.probability(wires=observable.wires)
            return self._dot(eigvals, prob)

        # estimate the ev
        return np.mean(self.sample(observable))

    @property
    def state(self):
        # returns the state after all operations are applied
        return self._state.dump()

    def reset(self):
        self._state.reset_all()

        # Dealloc copy before creating a second copy.
        del self._pre_rotated_state

        self._pre_rotated_state = QrackSimulator(cloneSid = self._state.sid)
