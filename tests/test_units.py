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
from pennylane_qrack.qrack_device import QrackDevice
from pyqrack import QrackSimulator


class TestDeviceUnits:
    """Unit tests for the plugin."""

    @pytest.mark.parametrize("num_wires, shots", [(1, None), (2, 184), (3, 1)])
    def test_device_attributes(self, num_wires, shots):
        """Test that attributes are set as expected."""
        dev = QrackDevice(wires=num_wires, shots=shots, isOpenCL=False)

        assert dev.num_wires == num_wires
        assert dev.shots == shots
        assert dev._samples is None
        assert dev._capabilities["model"] == "qubit"
        assert dev._capabilities["tensor_observables"]
        assert dev._capabilities["returns_state"]
        assert isinstance(dev._state, QrackSimulator)

    # def test_no_gpu_support(self, monkeypatch):
    #     """Test that error thrown when gpu set to True but no gpu support found."""
    #
    #     monkeypatch.setattr(QrackDevice, "gpu_supported", False, isOpenCL=False)
    #
    #     with pytest.raises(
    #         qml.DeviceError, match="GPU not supported with installed version of qrack"
    #     ):
    #         QrackDevice(3, gpu=True, isOpenCL=False)

    @pytest.mark.parametrize(
        "wires, prob",
        [
            ([0], [1.0, 0.0]),
            ([0, 1], [0.0, 1.0, 0.0, 0.0]),
            ([1, 3], [0.0, 0.0, 0.0, 1.0]),
        ],
    )
    def test_analytic_probability(self, wires, prob, tol):
        """Test the analytic_probability() function."""
        dev = QrackDevice(4, isOpenCL=False)
        state = np.array((0, 1, 0, 1))
        op = qml.BasisState(state, wires=[0, 1, 2, 3])
        dev.apply([op])

        res = dev.analytic_probability(wires=wires)
        res = list(res)
        assert np.allclose(res, prob, atol=tol)

    def test_reset(self, tol):
        """Test the reset() function."""
        dev = QrackDevice(4, isOpenCL=False)
        state = np.array((0, 1, 0, 1))
        op = qml.BasisState(state, wires=[0, 1, 2, 3])
        dev.apply([op])
        dev.reset()

        expected = [0.0] * 16
        expected[0] = 1.0
        actual = dev._state.out_ket()
        for i in range(16):
            actual[i] = actual[i] * np.conjugate(actual[i])
        assert np.allclose(actual, expected)
