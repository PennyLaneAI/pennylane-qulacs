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
from pennylane.ops import (
    QubitStateVector,
    BasisState,
    QubitUnitary,
    CRZ,
    PhaseShift,
    Adjoint,
)
from pennylane.wires import Wires

from pyqrack import QrackSimulator, Pauli

from ._version import __version__

# tolerance for numerical errors
tolerance = 1e-10


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
        "Prod": None,
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
    }

    config = pathlib.Path(
        os.path.dirname(sys.modules[__name__].__file__) + "/QrackDeviceConfig.toml"
    )

    @staticmethod
    def get_c_interface():
        return (
            "QrackDevice",
            os.path.dirname(sys.modules[__name__].__file__) + "/libqrack_device.so",
        )

    def __init__(self, wires=0, shots=None, **kwargs):
        super().__init__(wires=wires, shots=shots)
        self._state = QrackSimulator(self.num_wires, **kwargs)

    def _reverse_state(self):
        end = self.num_wires - 1
        mid = self.num_wires >> 1
        for i in range(mid):
            self._state.swap(i, end - i)

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
                        f"Operation {op.name} is not supported on a {self.short_name} device, except for single wires."
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

        self._reverse_state()
        if len(wires) != self.num_wires or sorted(wires, reverse=True) != wires:
            input_state = self._expand_state(input_state, wires)

        # call qrack' state initialization
        self._state.in_ket(input_state)
        self._reverse_state()

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

        par = op.parameters

        if opname == "MultiRZ":
            device_wires = self.map_wires(op.wires)
            for q in device_wires:
                self._state.r(Pauli.PauliZ, par[0], q)
            return

        if opname == "C(MultiRZ)":
            device_wires = self.map_wires(op.wires)
            control_wires = self.map_wires(op.control_wires)
            for q in device_wires:
                self._state.mcr(Pauli.PauliZ, par[0], control_wires, q)
            return

        # translate op wire labels to consecutive wire labels used by the device
        device_wires = self.map_wires(
            (op.control_wires + op.wires) if op.control_wires else op.wires
        )

        if opname in [
            "".join(p)
            for p in it.product(
                [
                    "Toffoli",
                    "C(Toffoli)",
                    "CNOT",
                    "C(CNOT)",
                    "MultiControlledX",
                    "C(PauliX)",
                ],
                ["", ".inv"],
            )
        ]:
            self._state.mcx(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["C(PauliY)", "C(PauliY).inv"]:
            self._state.mcy(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["C(PauliZ)", "C(PauliZ).inv"]:
            self._state.mcz(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["C(Hadamard)", "C(Hadamard).inv"]:
            self._state.mch(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in [
            "CSWAP",
            "CSWAP.inv",
            "C(SWAP)",
            "C(SWAP).inv",
            "C(CSWAP)",
            "C(CSWAP).inv",
        ]:
            self._state.cswap(
                device_wires.labels[:-2],
                device_wires.labels[-2],
                device_wires.labels[-1],
            )
        elif opname in ["CRX", "C(RX)", "C(CRX)"]:
            self._state.mcr(Pauli.PauliX, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRX.inv", "C(RX).inv", "C(CRX).inv"]:
            self._state.mcr(
                Pauli.PauliX, -par[0], device_wires.labels[:-1], device_wires.labels[-1]
            )
        elif opname in ["CRY", "C(RY)", "C(CRY)"]:
            self._state.mcr(Pauli.PauliY, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRY.inv", "C(RY).inv", "C(CRY).inv"]:
            self._state.mcr(
                Pauli.PauliY, -par[0], device_wires.labels[:-1], device_wires.labels[-1]
            )
        elif opname in ["CRZ", "C(RZ)", "C(CRZ)"]:
            self._state.mcr(Pauli.PauliZ, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRZ.inv", "C(RZ).inv", "C(CRZ).inv"]:
            self._state.mcr(
                Pauli.PauliZ, -par[0], device_wires.labels[:-1], device_wires.labels[-1]
            )
        elif opname in [
            "CRot",
            "CRot.inv",
            "C(Rot)",
            "C(Rot).inv",
            "C(CRot)",
            "C(CRot).inv",
        ]:
            phi = par[0]
            theta = par[1]
            omega = par[2]
            if ".inv" in opname:
                tmp = phi
                phi = -omega
                theta = -theta
                omega = -phi
            c = math.cos(theta / 2)
            s = math.sin(theta / 2)
            mtrx = [
                cmath.exp(-0.5j * (phi + omega)) * c,
                cmath.exp(0.5j * (phi - omega)) * s,
                cmath.exp(-0.5j * (phi - omega)) * s,
                cmath.exp(0.5j * (phi + omega)) * np.cos(theta / 2),
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
            self._state.cswap(
                device_wires.labels[:-2],
                device_wires.labels[-2],
                device_wires.labels[-1],
            )
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, 1j)
        elif opname == "C(ISWAP).inv":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, -1j)
            self._state.cswap(
                device_wires.labels[:-2],
                device_wires.labels[-2],
                device_wires.labels[-1],
            )
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, -1j)
        elif opname == "C(PSWAP)":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, par[0])
            self._state.cswap(
                device_wires.labels[:-2],
                device_wires.labels[-2],
                device_wires.labels[-1],
            )
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, par[0])
        elif opname == "C(PSWAP).inv":
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, -par[0])
            self._state.cswap(
                device_wires.labels[:-2],
                device_wires.labels[-2],
                device_wires.labels[-1],
            )
            self._state.mcu(device_wires.labels[:-1], device_wires.labels[-1], 0, 0, -par[0])
        elif opname in ["CY", "CY.inv", "C(CY)", "C(CY).inv"]:
            self._state.mcy(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CZ", "CZ.inv", "C(CZ)", "C(CZ).inv"]:
            self._state.mcz(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "S":
            for label in device_wires.labels:
                self._state.s(label)
        elif opname == "S.inv":
            for label in device_wires.labels:
                self._state.adjs(label)
        elif opname == "C(S)":
            self._state.mcs(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "C(S).inv":
            self._state.mcadjs(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "T":
            for label in device_wires.labels:
                self._state.t(label)
        elif opname == "T.inv":
            for label in device_wires.labels:
                self._state.adjt(label)
        elif opname == "C(T)":
            self._state.mct(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "C(T).inv":
            self._state.mcadjt(device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "RX":
            for label in device_wires.labels:
                self._state.r(Pauli.PauliX, par[0], label)
        elif opname == "RX.inv":
            for label in device_wires.labels:
                self._state.r(Pauli.PauliX, -par[0], label)
        elif opname in ["CRX", "C(RX)", "C(CRX)"]:
            self._state.mcr(Pauli.PauliX, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRX.inv", "C(RX).inv", "C(CRX).inv"]:
            self._state.mcr(Pauli.PauliX, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "RY":
            for label in device_wires.labels:
                self._state.r(Pauli.PauliY, par[0], label)
        elif opname == "RY.inv":
            for label in device_wires.labels:
                self._state.r(Pauli.PauliY, -par[0], label)
        elif opname in ["CRY", "C(RY)", "C(CRY)"]:
            self._state.mcr(Pauli.PauliY, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRY.inv", "C(RY).inv", "C(CRY).inv"]:
            self._state.mcr(Pauli.PauliY, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname == "RZ":
            for label in device_wires.labels:
                self._state.r(Pauli.PauliZ, par[0], label)
        elif opname == "RZ.inv":
            for label in device_wires.labels:
                self._state.r(Pauli.PauliZ, -par[0], label)
        elif opname in ["CRZ", "C(RZ)", "C(CRZ)"]:
            self._state.mcr(Pauli.PauliZ, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["CRZ.inv", "C(RZ).inv", "C(CRZ).inv"]:
            self._state.mcr(Pauli.PauliY, par[0], device_wires.labels[:-1], device_wires.labels[-1])
        elif opname in ["PauliX", "PauliX.inv"]:
            for label in device_wires.labels:
                self._state.x(label)
        elif opname in ["PauliY", "PauliY.inv"]:
            for label in device_wires.labels:
                self._state.y(label)
        elif opname in ["PauliZ", "PauliZ.inv"]:
            for label in device_wires.labels:
                self._state.z(label)
        elif opname in ["Hadamard", "Hadamard.inv"]:
            for label in device_wires.labels:
                self._state.h(label)
        elif opname == "SX":
            sx_mtrx = [(1 + 1j) / 2, (1 - 1j) / 2, (1 - 1j) / 2, (1 + 1j) / 2]
            for label in device_wires.labels:
                self._state.mtrx(sx_mtrx, label)
        elif opname == "SX.inv":
            isx_mtrx = [(1 - 1j) / 2, (1 + 1j) / 2, (1 + 1j) / 2, (1 - 1j) / 2]
            for label in device_wires.labels:
                self._state.mtrx(isx_mtrx, label)
        elif opname == "C(SX)":
            self._state.mcmtrx(
                device_wires.labels[:-1],
                [(1 + 1j) / 2, (1 - 1j) / 2, (1 - 1j) / 2, (1 + 1j) / 2],
                device_wires.labels[-1],
            )
        elif opname == "PhaseShift":
            p_mtrx = [1, 0, 0, cmath.exp(1j * par[0])]
            for label in device_wires.labels:
                self._state.mtrx(p_mtrx, label)
        elif opname == "PhaseShift.inv":
            ip_mtrx = [1, 0, 0, cmath.exp(1j * -par[0])]
            for label in device_wires.labels:
                self._state.mtrx(ip_mtrx, label)
        elif opname == "C(PhaseShift)":
            self._state.mtrx(
                device_wires.labels[:-1],
                [1, 0, 0, cmath.exp(1j * par[0])],
                device_wires.labels[-1],
            )
        elif opname == "C(PhaseShift).inv":
            self._state.mtrx(
                device_wires.labels[:-1],
                [1, 0, 0, cmath.exp(1j * -par[0])],
                device_wires.labels[-1],
            )
        elif opname in [
            "ControlledPhaseShift",
            "C(ControlledPhaseShift)",
            "CPhase",
            "C(CPhase)",
        ]:
            self._state.mcmtrx(
                device_wires.labels[:-1],
                [1, 0, 0, cmath.exp(1j * par[0])],
                device_wires.labels[-1],
            )
        elif opname in [
            "ControlledPhaseShift.inv",
            "C(ControlledPhaseShift).inv",
            "CPhase.inv",
            "C(CPhase).inv",
        ]:
            self._state.mcmtrx(
                device_wires.labels[:-1],
                [1, 0, 0, cmath.exp(1j * -par[0])],
                device_wires.labels[-1],
            )
        elif opname == "U3":
            for label in device_wires.labels:
                self._state.u(label, par[0], par[1], par[2])
        elif opname == "U3.inv":
            for label in device_wires.labels:
                self._state.u(label, -par[0], -par[2], -par[1])
        elif opname == "Rot":
            for label in device_wires.labels:
                self._state.r(Pauli.PauliZ, par[0], label)
                self._state.r(Pauli.PauliY, par[1], label)
                self._state.r(Pauli.PauliZ, par[2], label)
        elif opname == "Rot.inv":
            for label in device_wires.labels:
                self._state.r(Pauli.PauliZ, -par[2], label)
                self._state.r(Pauli.PauliY, -par[1], label)
                self._state.r(Pauli.PauliZ, -par[0], label)
        elif opname == "C(U3)":
            self._state.mcu(
                device_wires.labels[:-1],
                device_wires.labels[-1],
                par[0],
                par[1],
                par[2],
            )
        elif opname == "C(U3).inv":
            self._state.mcu(
                device_wires.labels[:-1],
                device_wires.labels[-1],
                -par[0],
                -par[2],
                -par[1],
            )
        elif opname not in [
            "Identity",
            "Identity.inv",
            "C(Identity)",
            "C(Identity).inv",
        ]:
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

        all_probs = self._abs(self.state) ** 2
        prob = self.marginal_prob(all_probs, wires)

        if (not "QRACK_FPPOW" in os.environ) or (6 > int(os.environ.get("QRACK_FPPOW"))):
            tot_prob = 0
            for p in prob:
                tot_prob = tot_prob + p

            if tot_prob != 1.0:
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

        if self.shots == 1:
            rev_sample = self._state.m_all()
            sample = 0
            for i in range(self.num_wires):
                if (rev_sample & (1 << i)) > 0:
                    sample |= 1 << (self.num_wires - (i + 1))
            self._samples = QubitDevice.states_to_binary(np.array([sample]), self.num_wires)

            return self._samples

        samples = np.array(
            self._state.measure_shots(list(range(self.num_wires - 1, -1, -1)), self.shots)
        )
        self._samples = QubitDevice.states_to_binary(samples, self.num_wires)

        return self._samples

    @property
    def state(self):
        # returns the state after all operations are applied
        self._reverse_state()
        o = self._state.out_ket()
        self._reverse_state()
        return o

    def reset(self):
        self._state.reset_all()
