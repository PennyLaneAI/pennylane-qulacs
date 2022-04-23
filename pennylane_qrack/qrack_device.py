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
from collections import OrderedDict
from functools import reduce
import cmath, math
import os
import itertools as it

import numpy as np

from pennylane import QubitDevice, DeviceError
from pennylane.ops import QubitStateVector, BasisState, QubitUnitary, CRZ, PhaseShift
from pennylane.wires import Wires

from pyqrack import QrackSimulator, Pauli

from . import __version__

# tolerance for numerical errors
tolerance = 1e-10


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

    def define_wire_map(self, wires):
        consecutive_wires = Wires(range(self.num_wires - 1, -1, -1))

        wire_map = zip(wires, consecutive_wires)
        return OrderedDict(wire_map)

    def apply(self, operations, **kwargs):
        rotations = kwargs.get("rotations", [])

        self.apply_operations(operations)

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
                raise DeviceError(
                    "Operation {} is not supported on a {} device.".format(op.name, self.short_name)
                )
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
        wires = op.wires
        input_state = op.parameters[0]

        if len(input_state) != 2 ** len(wires):
            raise ValueError("State vector must be of length 2**wires.")
        if input_state.ndim != 1 or len(input_state) != 2 ** len(wires):
            raise ValueError("State vector must be of length 2**wires.")
        if not np.isclose(np.linalg.norm(input_state, 2), 1.0, atol=tolerance):
            raise ValueError("Sum of amplitudes-squared does not equal one.")

        if len(wires) != self.num_wires or sorted(wires, reverse=True) != wires:
            input_state = self._expand_state(input_state, wires)

        # call qrack' state initialization
        self._state.in_ket(input_state)

    def _apply_basis_state(self, op):
        """Initialize a basis state"""
        wires = op.wires
        par = op.parameters[0]
        wire_count = len(wires)
        n_basis_state = len(par)

        if not set(par).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")
        if n_basis_state != wire_count:
            raise ValueError("BasisState parameter and wires must be of equal length.")

        for i in range(wire_count):
            index = (self.num_wires - 1) - wires.labels[i]
            if par[i] != self._state.m(index):
                self._state.x(index)

    def _apply_gate(self, op):
        """Apply native qrack gate"""

        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires(op.wires)
        par = op.parameters

        if op.name == "Toffoli" or op.name == "Toffoli.inv" or op.name == "CNOT" or op.name == "CNOT.inv":
            self._state.mcx(device_wires.labels[:-1], device_wires.labels[-1])
        elif op.name == "CSWAP" or op.name == "CSWAP.inv":
            self._state.cswap(device_wires.labels[:-2], device_wires.labels[-2], device_wires.labels[-1])
        elif op.name == "CRZ":
            self._state.mcr(Pauli.PauliZ, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif op.name == "CRZ.inv":
            self._state.mcr(Pauli.PauliZ, -par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif op.name == "SWAP" or op.name == "SWAP.inv":
            self._state.swap(device_wires.labels[0], device_wires.labels[1])
        elif op.name == "CZ" or op.name == "CZ.inv":
            self._state.mcz(device_wires.labels[:-1], device_wires.labels[-1])
        elif op.name == "S":
            self._state.s(device_wires.labels[0])
        elif op.name == "S.inv":
            self._state.adjs(device_wires.labels[0])
        elif op.name == "T":
            self._state.t(device_wires.labels[0])
        elif op.name == "T.inv":
            self._state.adjt(device_wires.labels[0])
        elif op.name == "RX":
            self._state.r(Pauli.PauliX, par[0], device_wires.labels[0])
        elif op.name == "RX.inv":
            self._state.r(Pauli.PauliX, -par[0], device_wires.labels[0])
        elif op.name == "RY":
            self._state.r(Pauli.PauliY, par[0], device_wires.labels[0])
        elif op.name == "RY.inv":
            self._state.r(Pauli.PauliY, -par[0], device_wires.labels[0])
        elif op.name == "RZ":
            self._state.r(Pauli.PauliZ, par[0], device_wires.labels[0])
        elif op.name == "RZ.inv":
            self._state.r(Pauli.PauliZ, -par[0], device_wires.labels[0])
        elif op.name == "PauliX" or op.name == "PauliX.inv":
            self._state.x(device_wires.labels[0])
        elif op.name == "PauliY" or op.name == "PauliY.inv":
            self._state.y(device_wires.labels[0])
        elif op.name == "PauliZ" or op.name == "PauliZ.inv":
            self._state.z(device_wires.labels[0])
        elif op.name == "Hadamard" or op.name == "Hadamard.inv":
            self._state.h(device_wires.labels[0])
        elif op.name == "PhaseShift":
            self._state.mtrx([1, 0, 0, cmath.exp(1j * par[0])], device_wires.labels[0])
        elif op.name == "PhaseShift.inv":
            self._state.mtrx([1, 0, 0, cmath.exp(1j * -par[0])], device_wires.labels[0])
        else:
            raise DeviceError(
                "Operation {} is not supported on a {} device.".format(op.name, self.short_name)
            )

    def analytic_probability(self, wires=None):
        """Return the (marginal) analytic probability of each computational basis state."""
        if self._state is None:
            return None

        all_probs = _reverse_state(self._abs(self.state) ** 2)
        prob = self.marginal_prob(all_probs, wires)

        if (not "QRACK_FPPOW" in os.environ) or (6 > int(os.environ.get('QRACK_FPPOW'))):
            tot_prob = 0
            for p in prob:
                tot_prob = tot_prob + p

            if tot_prob != 1.:
                for i in range(len(prob)):
                    prob[i] = prob[i] / tot_prob

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
                state_copy = QrackSimulator(cloneSid = self._state.sid)

                return state_copy.measure_pauli(observables, list(range(self.num_wires)))

            # exact expectation value
            eigvals = self._asarray(observable.eigvals, dtype=self.R_DTYPE)
            prob = self.probability(wires=observable.wires)
            return self._dot(eigvals, prob)

        # estimate the ev
        return np.mean(self.sample(observable))

    def generate_samples(self):
        if self.shots is None:
            raise qml.QuantumFunctionError(
                "The number of shots has to be explicitly set on the device "
                "when using sample-based measurements."
            )

        samples = np.array(self._state.measure_shots(list(range(self.num_wires - 1, -1, -1)), self.shots))
        self._samples = QubitDevice.states_to_binary(samples, self.num_wires)

        return self._samples

    @property
    def state(self):
        # returns the state after all operations are applied
        return self._state.out_ket()

    def reset(self):
        self._state.reset_all()
