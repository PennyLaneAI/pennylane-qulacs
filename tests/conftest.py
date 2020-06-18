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
import numpy as np
import pytest
import os

import pennylane as qml

from pennylane_qulacs.qulacs_device import QulacsDevice


np.random.seed(42)


# ==========================================================
# Some useful global variables

# single qubit unitary matrix
U = np.array([[0.83645892 - 0.40533293j, -0.20215326 + 0.30850569j],
              [-0.23889780 - 0.28101519j, -0.88031770 - 0.29832709j]])

# two qubit unitary matrix
U2 = np.array([[0, 1, 1, 1],
               [1, 0, 1, -1],
               [1, -1, 0, 1],
               [1, 1, -1, 0]]) / np.sqrt(3)

# single qubit Hermitian observable
A = np.array([[1.02789352, 1.61296440 - 0.3498192j],
              [1.61296440 + 0.3498192j, 1.23920938 + 0j]])


# ==========================================================
# pytest fixtures


TOL = 1e-3

@pytest.fixture(scope="session")
def tol():
    """Numerical tolerance for equality tests."""
    return float(os.environ.get("TOL", TOL))

@pytest.fixture
def init_state(scope="session"):
    """Fixture to create an n-qubit initial state"""
    def _init_state(n):
        state = np.random.random([2 ** n]) + np.random.random([2 ** n]) * 1j
        state /= np.linalg.norm(state)
        return state

    return _init_state


