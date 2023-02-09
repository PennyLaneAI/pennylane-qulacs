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


class TestDeviceUnits:
    """Unit tests for the plugin."""

    @pytest.mark.parametrize("num_wires, shots", [(1, None), (2, 184), (3, 1)])
    def test_device_attributes(self, num_wires, shots):
        """Test that attributes are set as expected."""
        dev = QulacsDevice(wires=num_wires, shots=shots)

        assert dev.num_wires == num_wires
        assert dev.shots == shots
        assert dev._samples is None
        assert dev._capabilities["model"] == "qubit"
        assert dev._capabilities["tensor_observables"]
        assert dev._capabilities["returns_state"]
        assert isinstance(dev._state, QuantumState)
        assert isinstance(dev._circuit, QuantumCircuit)

    def test_no_gpu_support(self, monkeypatch):
        """Test that error thrown when gpu set to True but no gpu support found."""

        monkeypatch.setattr(QulacsDevice, "gpu_supported", False)

        with pytest.raises(
            qml.DeviceError, match="GPU not supported with installed version of qulacs"
        ):
            QulacsDevice(3, gpu=True)

    @pytest.mark.parametrize(
        "wires, prob",
        [([0], [1.0, 0.0]), ([0, 1], [0.0, 1.0, 0.0, 0.0]), ([1, 3], [0.0, 0.0, 0.0, 1.0])],
    )
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

        expected = [0.0] * 16
        expected[0] = 1.0
        assert np.allclose(dev._state.get_vector(), expected)
        assert QuantumCircuit(4).calculate_depth() == 0

    @pytest.mark.parametrize(
        "obs,args,wires,supported",
        [
            (qml.PauliX, [], [0], True),
            (qml.Hadamard, [], [0], False),
            (
                qml.Hermitian,
                [
                    np.array(
                        [
                            [1.02789352, 1.61296440 - 0.3498192j],
                            [1.61296440 + 0.3498192j, 1.23920938 + 0j],
                        ]
                    )
                ],
                [0],
                False,
            ),
            (lambda wires: qml.PauliX(wires[0]) @ qml.Hadamard(wires[1]), [], [0, 1], False),
            (lambda wires: qml.PauliZ(wires[0]) @ qml.PauliY(wires[1]), [], [0, 1], True),
        ],
    )
    def test_expval_hadamard(self, obs, args, wires, supported, mocker):
        """Test that QulacsDevice.expval() uses native calculations when possible"""
        dev = QulacsDevice(4)

        spy = mocker.spy(dev, "probability")
        dev.expval(obs(*args, wires=wires))

        if supported:
            spy.assert_not_called()
        else:
            spy.assert_called_once()
