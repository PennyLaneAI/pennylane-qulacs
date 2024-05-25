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
import pathlib
import sys
import itertools as it

import numpy as np

from pennylane import QubitDevice, DeviceError
from pennylane.ops import QubitStateVector, BasisState, QubitUnitary, CRZ, PhaseShift, Adjoint
from pennylane.wires import Wires

from pyqrack import QrackSimulator, Pauli

from ._version import __version__

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

    _capabilities = {
        "model": "qubit",
        "tensor_observables": True,
        "inverse_operations": True,
        "returns_state": True,
    }

    _observable_map = {
        "PauliX": Pauli.PauliX,
        "PauliY": Pauli.PauliY,
        "PauliZ": Pauli.PauliZ,
        "Identity": Pauli.PauliI,
        "Prod": None
        # "Hadamard": None,
        # "Hermitian": None,
        # "Sum": None,
        # "SProd": None,
        # "Exp": None,
        # "Projector": None,
        # "Hamiltonian": None,
        # "SparseHamiltonian": None
    }

    observables = _observable_map.keys()
    operations = {
        "Identity",
        "C(Identity)",
        "MultiRZ",
        "C(MultiRZ)",
        "Toffoli",
        "C(Toffoli)",
        "CSWAP",
        "C(CSWAP)",
        "CRX",
        "C(CRX)",
        "CRY",
        "C(CRY)",
        "CRZ",
        "C(CRZ)",
        "CRot",
        "C(CRot)",
        "SWAP",
        "C(SWAP)",
        "ISWAP",
        "C(ISWAP)",
        "PSWAP",
        "C(PSWAP)",
        "CNOT",
        "C(CNOT)",
        "CY",
        "C(CY)",
        "CZ",
        "C(CZ)",
        "S",
        "C(S)",
        "T",
        "C(T)",
        "RX",
        "C(RX)",
        "RY",
        "C(RY)",
        "RZ",
        "C(RZ)",
        "PauliX",
        "C(PauliX)",
        "PauliY",
        "C(PauliY)",
        "PauliZ",
        "C(PauliZ)",
        "Hadamard",
        "C(Hadamard)",
        "SX",
        "C(SX)",
        "PhaseShift",
        "C(PhaseShift)",
        "U1",
        "C(U1)",
        "U2",
        "C(U2)",
        "U3",
        "C(U3)",
        "Rot",
        "C(Rot)",
        "ControlledPhaseShift",
        "CPhase",
        "C(ControlledPhaseShift)",
        "C(CPhase)",
        "MultiControlledX",
        "C(MultiControlledX)",
        "QFT"
    }

    config = pathlib.Path(os.path.dirname(sys.modules[__name__].__file__) + "/QrackDeviceConfig.toml")

    @staticmethod
    def get_c_interface():
        return "QrackDevice", os.path.dirname(sys.modules[__name__].__file__) + "/libqrack_device.so"

    def __init__(self, wires=0, shots=None, **kwargs):
        super().__init__(wires=wires, shots=shots)

        if "isTensorNetwork" in kwargs:
            self._state = QrackSimulator(self.num_wires, **kwargs)
        else:
            self._state = QrackSimulator(self.num_wires, isTensorNetwork=False, **kwargs)

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
            if isinstance(op, QubitStateVector):
                self._apply_qubit_state_vector(op)
            elif isinstance(op, BasisState):
                self._apply_basis_state(op)
            elif isinstance(op, QubitUnitary):
                if len(op.wires) > 1:
                    raise DeviceError(
                        "Operation {} is not supported on a {} device, except for single wires.".format(op.name, self.short_name)
                    )
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
        wires = self.map_wires(Wires(op.wires))
        par = op.parameters[0]
        wire_count = len(wires)
        n_basis_state = len(par)

        if not set(par).issubset({0, 1}):
            raise ValueError("BasisState parameter must consist of 0 or 1 integers.")
        if n_basis_state != wire_count:
            raise ValueError("BasisState parameter and wires must be of equal length.")

        for i in range(wire_count):
            index = wires.labels[i]
            if par[i] != self._state.m(index):
                self._state.x(index)

    def _apply_gate(self, op):
        """Apply native qrack gate"""

        opname = op.name
        if isinstance(op, Adjoint):
            op = op.base
            opname = op.name + ".inv"

        if opname == "MultiRZ":
            device_wires = self.map_wires(op.wires)
            for q in device_wires:
                self._state.r(Pauli.PauliZ, par[0], q)
        elif opname == "C(MultiRZ)":
            device_wires = self.map_wires(op.wires)
            control_wires = self.map_wires(op.control_wires)
            for q in device_wires:
                self._state.mcr(Pauli.PauliZ, par[0], control_wires, q)

        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires((op.control_wires + op.wires) if op.control_wires else op.wires)
        par = op.parameters

        if opname in [''.join(p) for p in product(["Toffoli", "C(Toffoli)", "CNOT", "C(CNOT)", "MultiControlledX", "C(PauliX)"], ["", ".inv"])]:
            self._state.mcx(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["C(PauliY)", "C(PauliY).inv"]:
            self._state.mcy(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["C(PauliZ)", "C(PauliZ).inv"]:
            self._state.mcz(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["C(Hadamard)", "C(Hadamard).inv"]:
            self._state.mch(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CSWAP", "CSWAP.inv", "C(SWAP)", "C(SWAP).inv", "C(CSWAP)", "C(CSWAP).inv"]:
            self._state.cswap(device_wires.labels[:-2], device_wires.labels[-2], device_wires.labels[-1])
        elif opname in ["CRX", "C(RX)", "C(CRX)"]:
            self._state.mcr(Pauli.PauliX, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRX.inv", "C(RX).inv", "C(CRX).inv"]:
            self._state.mcr(Pauli.PauliX, -par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRY", "C(RY)", "C(CRY)"]:
            self._state.mcr(Pauli.PauliY, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRY.inv", "C(RY).inv", "C(CRY).inv"]:
            self._state.mcr(Pauli.PauliY, -par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRZ", "C(RZ)", "C(CRZ)"]:
            self._state.mcr(Pauli.PauliZ, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRZ.inv", "C(RZ).inv", "C(CRZ).inv"]:
            self._state.mcr(Pauli.PauliZ, -par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRot", "CRot.inv", "C(Rot)", "C(Rot).inv", "C(CRot)", "C(CRot).inv"]:
            phi = par[0]
            theta = par[1]
            omega = par[2]
            if ".inv" in opname:
                phi = -phi
                theta = -theta
                omega = -omega
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            mtrx = [
                cmath.exp(-0.5j * (phi + omega)) * c, cmath.exp(0.5j * (phi - omega)) * s,
                cmath.exp(-0.5j * (phi - omega)) * s, cmath.exp(0.5j * (phi + omega)) * c
            ]
            self._state.mcmtrx(device_wires.labels[:-1], mtrx, device_wires.labels[-1])
        elif opname in ["SWAP", "SWAP.inv"]:
            self._state.swap(device_wires.labels[0], device_wires.labels[1])
        elif opname == "ISWAP":
            self._state.iswap(device_wires.labels[0], device_wires.labels[1])
        elif opname == "ISWAP.inv":
            self._state.adjiswap(device_wires.labels[0], device_wires.labels[1])
        elif opname == "C(ISWAP)":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, 1j)
            self._state.cswap(device_wires.labels[:-2], device_wires.labels[-2], device_wires.labels[-1])
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, 1j)
        elif opname == "C(ISWAP).inv":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, -1j)
            self._state.cswap(device_wires.labels[:-2], device_wires.labels[-2], device_wires.labels[-1])
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, -1j)
        elif opname == "C(PSWAP)":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, par[0])
            self._state.cswap(device_wires.labels[:-2], device_wires.labels[-2], device_wires.labels[-1])
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, par[0])
        elif opname == "C(PSWAP).inv":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, -par[0])
            self._state.cswap(device_wires.labels[:-2], device_wires.labels[-2], device_wires.labels[-1])
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, -par[0])
        elif opname in ["CY", "CY.inv", "C(CY)", "C(CY).inv"]:
            self._state.mcy(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CZ", "CZ.inv", "C(CZ)", "C(CZ).inv"]:
            self._state.mcz(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "S":
            self._state.s(device_wires.labels[0])
        elif opname == "S.inv":
            self._state.adjs(device_wires.labels[0])
        elif opname == "C(S)":
            self._state.mcs(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "C(S).inv":
            self._state.mcadjs(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "T":
            self._state.t(device_wires.labels[0])
        elif opname == "T.inv":
            self._state.adjt(device_wires.labels[0])
        elif opname == "C(T)":
            self._state.mct(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "C(T).inv":
            self._state.mcadjt(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "RX":
            self._state.r(Pauli.PauliX, par[0], device_wires.labels[0])
        elif opname == "RX.inv":
            self._state.r(Pauli.PauliX, -par[0], device_wires.labels[0])
        elif opname in ["CRX", "C(RX)", "C(CRX)"]:
            self._state.mcr(Pauli.PauliX, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRX.inv", "C(RX).inv", "C(CRX).inv"]:
            self._state.mcr(Pauli.PauliX, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "RY":
            self._state.r(Pauli.PauliY, par[0], device_wires.labels[0])
        elif opname == "RY.inv":
            self._state.r(Pauli.PauliY, -par[0], device_wires.labels[0])
        elif opname in ["CRY", "C(RY)", "C(CRY)"]:
            self._state.mcr(Pauli.PauliY, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRY.inv", "C(RY).inv", "C(CRY).inv"]:
            self._state.mcr(Pauli.PauliY, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "RZ":
            self._state.r(Pauli.PauliZ, par[0], device_wires.labels[0])
        elif opname == "RZ.inv":
            self._state.r(Pauli.PauliZ, -par[0], device_wires.labels[0])
        elif opname in ["CRZ", "C(RZ)", "C(CRZ)"]:
            self._state.mcr(Pauli.PauliZ, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRZ.inv", "C(RZ).inv", "C(CRZ).inv"]:
            self._state.mcr(Pauli.PauliY, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["PauliX", "PauliX.inv"]:
            self._state.x(device_wires.labels[0])
        elif opname in ["PauliY", "PauliY.inv"]:
            self._state.y(device_wires.labels[0])
        elif opname in ["PauliZ", "PauliZ.inv"]:
            self._state.z(device_wires.labels[0])
        elif opname in ["Hadamard", "Hadamard.inv"]:
            self._state.h(device_wires.labels[0])
        elif opname == "SX":
            self._state.mtrx([(1+1j)/2, (1-1j)/2, (1-1j)/2, (1+1j)/2], device_wires.labels[0])
        elif opname == "SX.inv":
            self._state.mtrx([(1-1j)/2, (1+1j)/2, (1+1j)/2, (1-1j)/2], device_wires.labels[0])
        elif opname == "C(SX)":
            self._state.mcmtrx(device_wires.labels[:-1], [(1+1j)/2, (1-1j)/2, (1-1j)/2, (1+1j)/2], device_wires.labels[-1])
        elif opname == "SX.inv":
            self._state.mtrx([(1-1j)/2, (1+1j)/2, (1+1j)/2, (1-1j)/2], device_wires.labels[0])
        elif opname == "C(SX).inc":
            self._state.mcmtrx(device_wires.labels[:-1], [(1-1j)/2, (1+1j)/2, (1+1j)/2, (1-1j)/2], device_wires.labels[-1])
        elif opname in ["PhaseShift", "U1"]:
            self._state.mtrx([1, 0, 0, cmath.exp(1j * par[0])], device_wires.labels[0])
        elif opname in ["PhaseShift.inv", "U1.inv"]:
            self._state.mtrx([1, 0, 0, cmath.exp(1j * -par[0])], device_wires.labels[0])
        elif opname in ["C(PhaseShift)", "C(U1)"]:
            self._state.mtrx(device_wires.labels[:-1], [1, 0, 0, cmath.exp(1j * par[0])], device_wires.labels[-1])
        elif opname in ["C(PhaseShift).inv", "C(U1).inv"]:
            self._state.mtrx(device_wires.labels[:-1], [1, 0, 0, cmath.exp(1j * -par[0])], device_wires.labels[-1])
        elif opname in ["ControlledPhaseShift", "C(ControlledPhaseShift)", "CPhase", "C(CPhase)"]:
            self._state.mcmtrx(device_wires.labels[:-1], [1, 0, 0, cmath.exp(1j * par[0])], device_wires.labels[-1])
        elif opname in ["ControlledPhaseShift.inv", "C(ControlledPhaseShift).inv", "CPhase.inv", "C(CPhase).inv"]:
            self._state.mcmtrx(device_wires.labels[:-1], [1, 0, 0, cmath.exp(1j * -par[0])], device_wires.labels[-1])
        elif opname == "U2":
            self._state.mtrx([1, cmath.exp(1j * par[1]), cmath.exp(1j * par[0]), cmath.exp(1j * (par[0] + par[1]))], device_wires.labels[0])
        elif opname == "U2.inv":
            self._state.mtrx([1, cmath.exp(1j * -par[1]), cmath.exp(1j * -par[0]), cmath.exp(1j * (-par[0] - par[1]))], device_wires.labels[0])
        elif opname == "C(U2)":
            self._state.mcmtrx(device_wires.labels[:-1], [1, cmath.exp(1j * par[1]), cmath.exp(1j * par[0]), cmath.exp(1j * (par[0] + par[1]))], device_wires.labels[-1])
        elif opname == "C(U2).inv":
            self._state.mcmtrx(device_wires.labels[:-1], [1, cmath.exp(1j * -par[1]), cmath.exp(1j * -par[0]), cmath.exp(1j * (-par[0] - par[1]))], device_wires.labels[-1])
        elif opname == "U3":
            self._state.u(device_wires.labels[-1], par[0], par[1], par[2])
        elif opname == "U3.inv":
            self._state.u(device_wires.labels[-1], -par[0], -par[1], -par[2])
        elif opname == "C(U3)":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], par[0], par[1], par[2])
        elif opname == "C(U3).inv":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], -par[0], -par[1], -par[2])
        elif opname == "QFT":
            self._state.qft(device_wires.labels)
            for i in range(len(wires) >> 1):
                self._state.swap(wires[i], wires[-i])
        elif opname == "QFT.inv":
            for i in range(len(wires) >> 1):
                self._state.swap(wires[i], wires[-i])
            self._state.iqft(device_wires.labels)
        elif opname not in ["Identity", "Identity.inv", "C(Identity)", "C(Identity).inv"]:
            raise DeviceError(f"Operation {opname} is not supported on a {self.short_name} device.")

    def _apply_qubit_unitary(self, op):
        """Apply unitary to state"""
        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires(op.wires)
        par = op.parameters

        if len(par[0]) != 2 ** len(device_wires):
            raise ValueError("Unitary matrix must be of shape (2**wires, 2**wires).")

        if isinstance(op, Adjoint):
            par[0] = par[0].conj().T

        matrix = par[0].flatten().tolist()
        self._state.mtrx(matrix, device_wires.labels[0])

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
                b = [self._observable_map[obs] for obs in observable.name]
            elif observable.name == "Prod":
                b = [self._observable_map[obs.name] for obs in observable.operands]
            else:
                b = [self._observable_map[observable.name]]

            if None not in b:
                q = self.map_wires(observable.wires)
                return self._state.pauli_expectation(q, b)

            # exact expectation value
            if callable(observable.eigvals):
                eigvals = self._asarray(observable.eigvals(), dtype=self.R_DTYPE)
            else:  # older version of pennylane
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