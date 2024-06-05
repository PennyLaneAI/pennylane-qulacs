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
"""Tests that device integrates with PennyLane"""
import numpy as np
import pennylane as qml
from pennylane_qrack.qrack_device import QrackDevice
from catalyst import qjit

class TestIntegration:
    """Some basic integration tests."""

    def test_load_device(self):
        """Test that the Qrack device loads correctly."""
        dev = QrackDevice(2, shots=int(1e6), isOpenCL=False)

        assert dev.num_wires == 2
        assert dev.shots == int(1e6)
        assert dev.short_name == "qrack.simulator"
        assert "model" in dev.__class__.capabilities()

    def test_expectation(self):
        """Test that expectation of a non-trivial circuit is correct."""
        dev = QrackDevice(2, shots=int(1e6), isOpenCL=False)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.adjoint(qml.RY(theta, wires=[0]))
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliX(wires=1))

        res = circuit()
        expected = np.array([np.sin(-theta) * np.sin(phi), np.sin(phi)])
        assert np.allclose(res, expected, atol=0.05)

    def test_expectation_qjit(self):
        """Test that expectation of a non-trivial circuit is correct."""
        dev = QrackDevice(2, shots=int(1e6), isOpenCL=False)

        theta = 0.432
        phi = 0.123

        @qjit
        @qml.qnode(dev)
        def circuit():
            qml.adjoint(qml.RY(theta, wires=[0]))
            qml.RY(phi, wires=[1])
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliX(wires=1))

        res = circuit()
        expected = np.array([np.sin(-theta) * np.sin(phi), np.sin(phi)])
        assert np.allclose(res, expected, atol=0.05)
