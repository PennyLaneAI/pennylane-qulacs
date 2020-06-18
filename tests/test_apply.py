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
from pennylane_qulacs.qulacs_device import QulacsDevice

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
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
CZ = np.diag([1, 1, 1, -1])
toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])
CSWAP = block_diag(I, I, SWAP)

# parametrized qubit gates
phase_shift = lambda phi: np.array([[1, 0], [0, np.exp(1j * phi)]])
rx = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * X
ry = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Y
rz = lambda theta: np.cos(theta / 2) * I + 1j * np.sin(-theta / 2) * Z
rot = lambda a, b, c: rz(c) @ (ry(b) @ rz(a))
crz = lambda theta: np.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, np.exp(-1j * theta / 2), 0],
        [0, 0, 0, np.exp(1j * theta / 2)],
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
    (qml.S(wires=0).inv(), S.conj().T),
    (qml.T(wires=0), T),
    (qml.T(wires=0).inv(), T.conj().T),
]

# list of all parametrized single-qubit gates
single_qubit_param = [
  # (qml.PhaseShift(0, wires=0), phase_shift),
    (qml.RX(0, wires=0), rx),
    (qml.RY(0, wires=0), ry),
    (qml.RZ(0, wires=0), rz)]
# list of all non-parametrized two-qubit gates
two_qubit = [
    (qml.CNOT(wires=[0, 1]), CNOT),
    (qml.SWAP(wires=[0, 1]), SWAP),
    #(qml.CZ(wires=[0, 1]), CZ)
]
# list of all parametrized two-qubit gates
two_qubit_param = [
#    (qml.CRZ(0, wires=[0, 1]), crz)
]
# list of all three-qubit gates
three_qubit = [
#    (qml.Toffoli(wires=[0, 1, 2]), toffoli),
#    (qml.CSWAP(wires=[0, 1, 2]), CSWAP)
    ]


def reverse_state(state):
    state = np.array(state)
    N = int(np.log2(len(state)))
    return state.reshape([2] * N).T.flatten()


class TestStateApply:
    """Test the device's state after application of gates."""

    @pytest.mark.parametrize("state", [np.array([0, 0, 1, 0]),
                                       np.array([0, 0, 1, 0]),
                                       np.array([1, 0, 1, 0]),
                                       np.array([1, 1, 1, 1])]
                             )
    def test_basis_state(self, state, tol):
        """Test basis state initialization"""
        dev = QulacsDevice(4)

        op = qml.BasisState(state, wires=[0, 1, 2, 3])
        dev.apply([op])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        # compute expected probabilities
        expected = np.zeros([2 ** 4])
        expected[np.ravel_multi_index(state, [2] * 4)] = 1

        assert np.allclose(res, expected, tol)

    def test_qubit_state_vector(self, init_state, tol):
        """Test QubitStateVector application"""
        dev = QulacsDevice(1)
        state = init_state(1)

        op = qml.QubitStateVector(state, wires=[0])
        dev.apply([op])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(state) ** 2
        assert np.allclose(res, expected, tol)

    def test_invalid_qubit_state_vector(self):
        """Test that an exception is raised if the state
        vector is the wrong size"""
        dev = QulacsDevice(2)
        state = np.array([0, 123.432])

        with pytest.raises(ValueError, match=r"State vector must be of length 2\*\*wires"):
            op = qml.QubitStateVector(state, wires=[0, 1])
            dev.apply([op])

    @pytest.mark.parametrize("op,mat", single_qubit)
    def test_single_qubit_no_parameters(self, init_state, op, mat, tol):
        """Test PauliX application"""
        dev = QulacsDevice(1)
        state = init_state(1)

        dev.apply([qml.QubitStateVector(state, wires=[0]), op])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(mat @ state) ** 2
        assert np.allclose(res, expected, tol)

    @pytest.mark.parametrize("theta", [0.5432, -0.232])
    @pytest.mark.parametrize("op,func", single_qubit_param)
    def test_single_qubit_parameters(self, init_state, op, func, theta, tol):
        """Test PauliX application"""
        dev = QulacsDevice(1)
        state = init_state(1)

        op.params = [theta]
        dev.apply([qml.QubitStateVector(state, wires=[0]), op])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(func(theta) @ state) ** 2
        assert np.allclose(res, expected, tol)

    def test_rotation(self, init_state, tol):
        """Test three axis rotation gate"""
        dev = QulacsDevice(1)
        state = init_state(1)

        a = 0.542
        b = 1.3432
        c = -0.654

        op = qml.Rot(a, b, c, wires=0)
        dev.apply([qml.QubitStateVector(state, wires=[0]), op])
        dev._obs_queue = []
        dev.pre_measure()

        res = np.abs(dev.state) ** 2
        expected = np.abs(rot(a, b, c) @ state) ** 2

        assert np.allclose(res, expected, tol)

    # @pytest.mark.parametrize("op,mat", two_qubit)
    # def test_two_qubit_no_parameters(self, init_state, op, mat, tol):
    #     """Test PauliX application"""
    #     dev = QulacsDevice(2)
    #     state = init_state(2)
    #
    #     dev.apply([qml.QubitStateVector(state, wires=[0, 1]), op])
    #     dev._obs_queue = []
    #     dev.pre_measure()
    #
    #     res = np.abs(reverse_state(dev.state)) ** 2
    #     expected = np.abs(mat @ state) ** 2
    #     assert np.allclose(res, expected, tol)

    # # @pytest.mark.parametrize("mat", [U, U2])
    # # def test_qubit_unitary(self, init_state, mat, tol):
    # #     """Test QubitUnitary application"""
    # #
    # #     N = int(np.log2(len(mat)))
    # #     dev = QulacsDevice(N)
    # #     state = init_state(N)
    # #
    # #     op = qml.QubitUnitary(mat, wires=list(range(N)))
    # #     dev.apply(qml.QubitStateVector(state, wires=list(range(N))))
    # #     dev.apply(op)
    # #     dev._obs_queue = []
    # #     dev.pre_measure()
    # #
    # #     res = np.abs(dev.state) ** 2
    # #     expected = np.abs(mat @ state) ** 2
    # #     assert np.allclose(res, expected, tol)
    #
    # def test_invalid_qubit_state_unitary(self):
    #     """Test that an exception is raised if the
    #     unitary matrix is the wrong size"""
    #     dev = QulacsDevice(2)
    #     state = np.array([[0, 123.432], [-0.432, 023.4]])
    #     op = qml.QubitUnitary(state, wires=[0, 1])
    #
    #     with pytest.raises(ValueError, match=r"Unitary matrix must be of shape"):
    #         dev.apply([op])
    #
    # @pytest.mark.parametrize("op, mat", three_qubit)
    # def test_three_qubit_no_parameters(self, init_state, op, mat, tol):
    #     dev = QulacsDevice(3)
    #     state = init_state(3)
    #
    #     dev.apply([qml.QubitStateVector(state, wires=[0, 1, 2]), op])
    #     dev._obs_queue = []
    #     dev.pre_measure()
    #
    #     res = np.abs(dev.state) ** 2
    #     expected = np.abs(mat @ state) ** 2
    #     assert np.allclose(res, expected, tol)
    #
    # @pytest.mark.parametrize("theta", [0.5432, -0.232])
    # @pytest.mark.parametrize("op,func", two_qubit_param)
    # def test_two_qubit_parameters(self, init_state, op, func, theta, tol):
    #     """Test parametrized two qubit gates application"""
    #     dev = QulacsDevice(2)
    #     state = init_state(2)
    #
    #     op.params = [theta]
    #     dev.apply([qml.QubitStateVector(state, wires=[0, 1]), op])
    #
    #     dev._obs_queue = []
    #     dev.pre_measure()
    #
    #     res = np.abs(dev.state) ** 2
    #     expected = np.abs(func(theta) @ state) ** 2
    #     assert np.allclose(res, expected, **tol)

