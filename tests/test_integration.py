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
from pennylane_qulacs.qulacs_device import QulacsDevice


class TestIntegration:
    """Some basic integration tests."""

    def test_load_device(self):
        """Test that the Qulacs device loads correctly."""
        dev = QulacsDevice(2, shots=2984)

        assert dev.num_wires == 2
        assert dev.shots == 2984
        assert dev.short_name == 'qulacs.simulator'
        assert "model" in dev.__class__.capabilities()

    def test_expectation(self):
        """Test that expectation of a non-trivial circuit is correct."""
        dev = QulacsDevice(2, shots=2984)

        theta = 0.432
        phi = 0.123

        @qml.qnode(dev)
        def circuit():
            qml.RX(theta, wires=[0])
            qml.RX(phi, wires=[1]).inv()
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliX(wires=0)), qml.expval(qml.PauliZ(wires=1))

        res = circuit()
        assert np.allclose(
            res, np.array([np.cos(theta), np.cos(theta) * np.cos(-phi)]), atol=1e-8
        )
