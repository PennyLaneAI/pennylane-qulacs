# Copyright 2018 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests that application of operations works correctly in the plugin devices"""
import pytest

import numpy as np
import pennylane as qml
from scipy.linalg import block_diag
from pennylane_qrack.qrack_device import QrackDevice

from conftest import U, U2, A

np.random.seed(42)


# ==========================================================
# Some useful global variables

# non-parametrized qubit gates
I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
S = np.diag([1, 1j])
T = np.diag([1, np.exp(1j * np.pi / 4)])
SX = np.array([[(1+1j)/2, (1-1j)/2], [(1-1j)/2, (1+1j)/2]])
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
ISWAP = np.array([[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]])
toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
multix4 = np.diag([1 for i in range(16)])
multix4[14:16, 14:16] = np.array([[0, 1], [1, 0]])
CSWAP = block_diag(I, I, SWAP)

# parametrized qubit gates
phase_shift = lambda phi: np.diag([1, np.exp(1j * phi)])
c_phase_shift = lambda phi: np.diag([1, 1, 1, np.exp(1j * phi)])
rx = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X
ry = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y
rz = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z
u2 = lambda phi, delta: np.array([[1, np.exp(1j * delta)], [np.exp(1j * phi), np.exp(1j * (phi + delta))]])
u3 = lambda theta, phi, delta: np.array([[np.cos(theta / 2), -np.exp(1j * delta) * np.sin(theta / 2)], [np.exp(1j * phi) * np.sin(theta / 2), np.exp(1j * (phi + delta)) * np.cos(theta / 2)]])

# CRX/CRY/CRZ in the PennyLane convention
crx = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.cos(theta / 2), -1j * np.sin(theta / 2)],
        [0, 0, -1j * np.sin(theta / 2), np.cos(theta / 2)],
    ]
)
cry = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.cos(theta / 2), -np.sin(theta / 2)],
        [0, 0, np.sin(theta / 2), np.cos(theta / 2)],
    ]
)
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
    ]
)
crot = lambda phi, theta, omega: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-0.5j * (phi + omega)) * np.cos(theta / 2), np.exp(0.5j * (phi - omega)) * np.sin(theta / 2)],
        [0, 0, np.exp(-0.5j * (phi - omega)) * np.sin(theta / 2), np.exp(0.5j * (phi + omega)) * np.cos(theta / 2)]
    ]
)

# list of all non-parametrized single-qubit gates,
# along with the PennyLane operation name
single_qubit = [
    (qml.PauliX(wires=0), X),
    (qml.PauliY(wires=0), Y),
    (qml.PauliZ(wires=0), Z),
    (qml.Hadamard(wires=0), H),
    (qml.S(wires=0), S),
    (qml.T(wires=0), T),
    (qml.SX(wires=0), SX),
    (qml.adjoint(qml.PauliX(wires=0)), X.conj().T),
    (qml.adjoint(qml.PauliY(wires=0)), Y.conj().T),
    (qml.adjoint(qml.PauliZ(wires=0)), Z.conj().T),
    (qml.adjoint(qml.Hadamard(wires=0)), H.conj().T),
    (qml.adjoint(qml.S(wires=0)), S.conj().T),
    (qml.adjoint(qml.T(wires=0)), T.conj().T),
    (qml.adjoint(qml.SX(wires=0)), SX.conj().T)
]

# list of all parametrized single-qubit gates
single_qubit_param = [
    (qml.RX(0, wires=0), rx),
    (qml.RY(0, wires=0), ry),
    (qml.RZ(0, wires=0), rz),
    (qml.PhaseShift(0, wires=0), phase_shift),
    (qml.adjoint(qml.RX(0, wires=0)), lambda theta: rx(-theta)),
    (qml.adjoint(qml.RY(0, wires=0)), lambda theta: ry(-theta)),
    (qml.adjoint(qml.RZ(0, wires=0)), lambda theta: rz(-theta)),
    (qml.adjoint(qml.PhaseShift(0, wires=0)), lambda theta: phase_shift(-theta)),
]
single_qubit_two_param = [
    (qml.U2(0, 0, wires=0), u2),
    (qml.adjoint(qml.U2(0, 0, wires=0)), lambda phi, delta: u2(-phi, -delta)),
]
single_qubit_three_param = [
    (qml.U3(0, 0, 0, wires=0), u3),
    (qml.adjoint(qml.U3(0, 0, 0, wires=0)), lambda theta, phi, delta: u3(-theta, -phi, -delta)),
]
# list of all non-parametrized two-qubit gates
two_qubit = [
    (qml.CNOT(wires=[0, 1]), CNOT),
    (qml.SWAP(wires=[0, 1]), SWAP),
    (qml.CZ(wires=[0, 1]), CZ),
    (qml.ISWAP(wires=[0, 1]), ISWAP),
    (qml.adjoint(qml.CNOT(wires=[0, 1])), CNOT.conj().T),
    (qml.adjoint(qml.SWAP(wires=[0, 1])), SWAP.conj().T),
    (qml.adjoint(qml.CZ(wires=[0, 1])), CZ.conj().T),
    (qml.adjoint(qml.ISWAP(wires=[0, 1])), ISWAP.conj().T)
]
# list of all parametrized two-qubit gates
two_qubit_param = [
    (qml.CRX(0, wires=[0, 1]), crx),
    (qml.CRY(0, wires=[0, 1]), cry),
    (qml.CRZ(0, wires=[0, 1]), crz),
    (qml.ControlledPhaseShift(0, wires=[0, 1]), c_phase_shift),
    (qml.adjoint(qml.CRX(0, wires=[0, 1])), lambda theta: crx(-theta)),
    (qml.adjoint(qml.CRY(0, wires=[0, 1])), lambda theta: cry(-theta)),
    (qml.adjoint(qml.CRZ(0, wires=[0, 1])), lambda theta: crz(-theta)),
    (qml.adjoint(qml.ControlledPhaseShift(0, wires=[0, 1])), lambda theta: c_phase_shift(-theta))
]
two_qubit_three_param = [
    (qml.CRot(0, 0, 0, wires=[0, 1]), crot),
    (qml.adjoint(qml.CRot(0, 0, 0, wires=[0, 1])), lambda phi, theta, omega: crot(-phi, -theta, -omega)),
]
# list of all three-qubit gates
three_qubit = [
    (qml.Toffoli(wires=[0, 1, 2]), toffoli),
    (qml.CSWAP(wires=[0, 1, 2]), CSWAP),
    (qml.adjoint(qml.Toffoli(wires=[0, 1, 2])), toffoli.conj().T),
    (qml.adjoint(qml.CSWAP(wires=[0, 1, 2])), CSWAP.conj().T),
]
# list of all four-qubit gates
four_qubit = [
    (qml.MultiControlledX(wires=[0, 1, 2, 3]), multix4),
    (qml.adjoint(qml.MultiControlledX(wires=[0, 1, 2, 3])), multix4.conj().T)
]


class TestStateApply:
    """Test the device's state after application of gates."""

    @pytest.mark.parametrize(
        "state",
        [
            np.array([0, 0, 1, 0]),
            np.array([0, 0, 1, 0]),
            np.array([1, 0, 1, 0]),
            np.array([1, 1, 1, 1]),
        ],
    )
    def test_basis_state(self, state, tol):
        """Test basis state initialization"""
        dev = QrackDevice(4)

        op = qml.BasisState(state, wires=[0, 1, 2, 3])
        dev.apply([op])
        dev._obs_queue = []

        res = np.abs(dev.state) ** 2
        # compute expected probabilities
        expected = np.zeros([2 ** 4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1

        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize(
        "state",
        [
            np.array([0, 0]),
            np.array([1, 0]),
            np.array([0, 1]),
            np.array([1, 1]),
        ],
    )
    @pytest.mark.parametrize("device_wires", [3, 4, 5])
    @pytest.mark.parametrize("op_wires", [[0, 1], [1, 0], [2, 0]])
    def test_basis_state_on_wires_subset(self, state, device_wires, op_wires, tol):
        """Test basis state initialization on a subset of device wires"""
        dev = QrackDevice(device_wires)

        op = qml.BasisState(state, wires=op_wires)
        dev.apply([op])
        dev._obs_queue = []

        res = np.abs(dev.state) ** 2
        # compute expected probabilities
        expected = np.zeros([2 ** len(op_wires)])
        expected[np.ravel_multi_index(state, [2] * len(op_wires))] = 1

        expected = dev._expand_state(expected, op_wires)
        assert np.allclose(res, expected, tol)

    def test_qubit_state_vector(self, init_state, tol):
        """Test QubitStateVector application"""
        dev = QrackDevice(1)
        state = init_state(1)

        op = qml.QubitStateVector(state, wires=[0])
        dev.apply([op])
        dev._obs_queue = []

        res = dev.state
        expected = state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("device_wires", [3, 4, 5])
    @pytest.mark.parametrize("op_wires", [[0], [2], [0, 1], [1, 0], [2, 0]])
    def test_qubit_state_vector_on_wires_subset(self, init_state, device_wires, op_wires, tol):
        """Test QubitStateVector application on a subset of device wires"""
        dev = QrackDevice(device_wires)
        state = init_state(len(op_wires))

        op = qml.QubitStateVector(state, wires=op_wires)
        dev.apply([op])
        dev._obs_queue = []

        res = dev.state
        expected = dev._expand_state(state, op_wires)

        assert np.allclose(res, expected, tol)

    def test_invalid_qubit_state_vector(self):
        """Test that an exception is raised if the state
        vector is the wrong size"""
        dev = QrackDevice(2)
        state = np.array([0, 123.432])

        with pytest.raises(ValueError, match=r"State vector must be of length 2\*\*wires"):
            op = qml.QubitStateVector(state, wires=[0, 1])
            dev.apply([op])

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test PauliX application"""
        dev = QrackDevice(1)
        state = init_state(1)

        dev.apply([qml.QubitStateVector(state, wires=[0]), op])
        dev._obs_queue = []

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test PauliX application"""
        dev = QrackDevice(1)
        state = init_state(1)

        op.data = [theta]
        dev.apply([qml.QubitStateVector(state, wires=[0]), op])
        dev._obs_queue = []

        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("phi", [0.126, -0.721])
    @pytest.mark.parametrize("delta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_two_param)
    def test_single_qubit_two_parameters(self, init_state, op, func, phi, delta, tol):
        """Test PauliX application"""
        dev = QrackDevice(1)
        state = init_state(1)

        op.data = [phi, delta]
        dev.apply([qml.QubitStateVector(state, wires=[0]), op])
        dev._obs_queue = []

        res = dev.state
        expected = func(phi, delta) @ state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("phi", [0.126, -0.721])
    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("omega", [1.213, -0.221])
    @pytest.mark.parametrize("op,func", single_qubit_three_param)
    def test_single_qubit_three_parameters(self, init_state, op, func, phi, theta, omega, tol):
        """Test PauliX application"""
        dev = QrackDevice(1)
        state = init_state(1)

        op.data = [phi, theta, omega]
        dev.apply([qml.QubitStateVector(state, wires=[0]), op])
        dev._obs_queue = []

        res = dev.state
        expected = func(phi, theta, omega) @ state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("op, mat", two_qubit)
    def test_two_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test PauliX application"""
        dev = QrackDevice(2)
        state = init_state(2)

        dev.apply([qml.QubitStateVector(state, wires=[0, 1]), op])
        dev._obs_queue = []

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("mat", [U])
    def test_qubit_unitary(self, init_state, mat, tol):
        """Test QubitUnitary application"""

        N = int(np.log2(len(mat)))
        dev = QrackDevice(N)
        state = init_state(N)

        op = qml.QubitUnitary(mat, wires=list(range(N)))
        dev.apply([qml.QubitStateVector(state, wires=list(range(N))), op])
        dev._obs_queue = []

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, tol)

    def test_invalid_qubit_state_unitary(self):
        """Test that an exception is raised if the
        unitary matrix is the wrong size"""
        state = np.array([[0, 123.432], [-0.432, 023.4]])

        with pytest.raises(ValueError, match=r"Input unitary must be of shape"):
            qml.QubitUnitary(state, wires=[0, 1])

    @pytest.mark.parametrize("op, mat", three_qubit)
    def test_three_qubit_no_parameters(self, init_state, op, mat, tol):
        dev = QrackDevice(3)
        state = init_state(3)

        dev.apply([qml.QubitStateVector(state, wires=[0, 1, 2]), op])
        dev._obs_queue = []

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("op, mat", four_qubit)
    def test_four_qubit_no_parameters(self, init_state, op, mat, tol):
        dev = QrackDevice(4)
        state = init_state(4)

        dev.apply([qml.QubitStateVector(state, wires=[0, 1, 2, 3]), op])
        dev._obs_queue = []

        res = dev.state
        expected = mat @ state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", two_qubit_param)
    def test_two_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test parametrized two qubit gates application"""
        dev = QrackDevice(2)
        state = init_state(2)

        op.data = [theta]
        dev.apply([qml.QubitStateVector(state, wires=[0, 1]), op])

        dev._obs_queue = []

        res = dev.state
        expected = func(theta) @ state
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("phi", [0.126, -0.721])
    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("omega", [1.213, -0.221])
    @pytest.mark.parametrize("op,func", two_qubit_three_param)
    def test_two_qubit_three_parameters(self, init_state, op, func, phi, theta, omega, tol):
        """Test parametrized two qubit gates application"""
        dev = QrackDevice(2)
        state = init_state(2)

        op.data = [phi, theta, omega]
        dev.apply([qml.QubitStateVector(state, wires=[0, 1]), op])

        dev._obs_queue = []

        res = dev.state
        expected = func(phi, theta, omega) @ state
        assert np.allclose(res, expected, tol)

    def test_apply_errors_qubit_state_vector(self):
        """Test that apply fails for incorrect state preparation."""
        dev = QrackDevice(2)

        with pytest.raises(ValueError, match="Sum of amplitudes-squared does not equal one."):
            dev.apply([qml.QubitStateVector(np.array([1, -1]), wires=[0])])

        with pytest.raises(ValueError, match=r"State vector must be of length 2\*\*wires."):
            p = np.array([1, 0, 1, 1, 0]) / np.sqrt(3)
            dev.reset()
            dev.apply([qml.QubitStateVector(p, wires=[0, 1])])

    def test_apply_errors_basis_state(self):
        """Test that apply fails for incorrect basis state preparation."""
        dev = QrackDevice(2)

        with pytest.raises(
            ValueError, match="BasisState parameter must consist of 0 or 1 integers."
        ):
            dev.apply([qml.BasisState(np.array([-0.2, 4.2]), wires=[0, 1])])

        with pytest.raises(
            ValueError, match="BasisState parameter and wires must be of equal length."
        ):
            dev.apply([qml.BasisState(np.array([0, 1]), wires=[0])])


@pytest.mark.parametrize(
    "state, device_wires, op_wires, expected",
    [
        (np.array([1, 0]), 2, [0], [1, 0, 0, 0]),
        (np.array([0, 1]), 2, [0], [0, 0, 1, 0]),
        (np.array([1, 1]) / np.sqrt(2), 2, [1], np.array([1, 1, 0, 0]) / np.sqrt(2)),
        (np.array([1, 1]) / np.sqrt(2), 3, [0], np.array([1, 0, 0, 0, 1, 0, 0, 0]) / np.sqrt(2)),
        (np.array([1, 2, 3, 4]) / np.sqrt(48), 3, [0, 1], np.array([1, 0, 2, 0, 3, 0, 4, 0]) / np.sqrt(48)),
        (np.array([1, 2, 3, 4]) / np.sqrt(48), 3, [1, 0], np.array([1, 0, 3, 0, 2, 0, 4, 0]) / np.sqrt(48)),
        (np.array([1, 2, 3, 4]) / np.sqrt(48), 3, [0, 2], np.array([1, 2, 0, 0, 3, 4, 0, 0]) / np.sqrt(48)),
        (np.array([1, 2, 3, 4]) / np.sqrt(48), 3, [1, 2], np.array([1, 2, 3, 4, 0, 0, 0, 0]) / np.sqrt(48)),
    ],
)
def test_expand_state(state, op_wires, device_wires, expected, tol):
    """Test that the expand_state method works as expected."""
    dev = QrackDevice(device_wires)
    res = dev._expand_state(state, op_wires)

    assert np.allclose(res, expected, tol)
