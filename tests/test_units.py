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
from pennylane_qulacs.qulacs_device import QulacsDevice
from qulacs import QuantumState, QuantumCircuit


class TestUnits:
    """Unit tests for the plugin."""

    @pytest.mark.parametrize("num_wires, shots, analytic",
                             [(1, 50, True),
                              (2, 184, False),
                              (3, 1, True)])
    def test_device_attributes(self, num_wires, shots, analytic):
        """Test that attributes are set as expected."""
        dev = QulacsDevice(wires=num_wires, shots=shots, analytic=analytic)

        assert dev.num_wires == num_wires
        assert dev.shots == shots
        assert dev.analytic == analytic
        assert dev._samples is None
        assert dev.circuit_hash is None
        assert dev._capabilities["model"] == "qubit"
        assert dev._capabilities["tensor_observables"]
        assert dev._capabilities["inverse_operations"]
        assert isinstance(dev._state, QuantumState)
        assert isinstance(dev._circuit, QuantumCircuit)

    def test_no_gpu_support(self, monkeypatch):
        """Test that error thrown when gpu set to True but no gpu support found."""

        monkeypatch.setattr(QulacsDevice, "gpu_supported", False)

        with pytest.raises(qml.DeviceError, match="GPU not supported with installed version of qulacs"):
            QulacsDevice(3, gpu=True)

    @pytest.mark.parametrize("wires, prob", [([0], [1., 0.]),
                                             ([0, 1], [0., 1., 0., 0.]),
                                             ([1, 3], [0., 0., 0., 1.])])
    def test_analytic_probability(self, wires, prob, tol):
        """Test the analytic_probability() function."""
        dev = QulacsDevice(4)
        state = np.array((0, 1, 0, 1))
        op = qml.BasisState(state, wires=[0, 1, 2, 3])
        dev.apply([op])

        res = dev.analytic_probability(wires=wires)
        res = list(res)
        assert np.allclose(res, prob, atol=tol)

    def test_reset(self, tol):
        """Test the reset() function."""
        dev = QulacsDevice(4)
        state = np.array((0, 1, 0, 1))
        op = qml.BasisState(state, wires=[0, 1, 2, 3])
        dev.apply([op])
        dev.reset()

        expected = [0.]*16
        expected[0] = 1.
        assert np.allclose(dev._state.get_vector(), expected)
        assert QuantumCircuit(4).calculate_depth() == 0


