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
"""Tests for any plugin- or framework-specific behaviour of the plugin devices"""
import pytest

import numpy as np

#from pennylane_qulacs.qulacs_device import z_eigs
from pennylane_qulacs import QulacsDevice


Z = np.diag([1, -1])



class TestProbabilities:
    """Tests for the probability function"""

    def test_probability_no_results(self):
        """Test that the probabilities function returns
        None if no job has yet been run."""
        dev = Device1(backend="statevector_simulator", wires=1, shots=0)
        assert dev.probabilities() is None
