import numpy as np
import functools
import itertools
from scipy.linalg import block_diag
from collections import OrderedDict

from pennylane import Device, DeviceError

import qulacs.gate as gate
from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.state import inner_product

from . import __version__


GPU_SUPPORTED = True
try:
    from qulacs import QuantumStateGpu
except ImportError:
    GPU_SUPPORTED = False


I = np.identity(2)
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
H = np.array([[1, 1], [1, -1]])/np.sqrt(2)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
CSWAP = block_diag(I, I, SWAP)

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
toffoli = np.diag([1 for i in range(8)])
toffoli[6:8, 6:8] = np.array([[0, 1], [1, 0]])


def hermitian(*args):
    r"""Input validation for an arbitary Hermitian expectation.
    Args:
        args (array): square hermitian matrix
    Returns:
        array: square hermitian matrix
    """
    A = np.asarray(args[0])

    if A.shape[0] != A.shape[1]:
        raise ValueError('Expectation must be a square matrix.')

    if not np.allclose(A, A.conj().T):
        raise ValueError('Expectation must be Hermitian.')

    return A


class QulacsDevice(Device):
    """Qulacs device"""
    name = 'Qulacs device'
    short_name = 'qulacs.simulator'
    pennylane_requires = '>=0.5.0'
    version = __version__
    author = 'Steven Oud'

    _capabilities = {
        'model': 'qubit',
        'tensor_observables': True
    }

    operations = {'CNOT', 'RX', 'RY', 'RZ', 'Rot', 'QubitStateVector',
                  'PauliX', 'PauliY', 'PauliZ', 'Hadamard'}
    observables = {'PauliX', 'PauliY', 'PauliZ', 'Hermitian'}

    _operations_map = {
        'QubitStateVector': None,
        'BasisState': None,
        'QubitUnitary': None,
        'Toffoli': toffoli,
        'CSWAP': CSWAP,
        'CRZ': crz,
        'Rot': rot,
        'SWAP': gate.SWAP,
        'CNOT': gate.CNOT,
        'CZ': gate.CZ,
        'S': gate.S,
        'Sdg': gate.Sdag,
        'T': gate.T,
        'Tdg': gate.Tdag,
        'RX': gate.RX,
        'RY': gate.RY,
        'RZ': gate.RZ,
        'PauliX': gate.X,
        'PauliY': gate.Y,
        'PauliZ': gate.Z,
        'Hadamard': gate.H
    }
    _observable_map = {
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hadamard': H,
        'Identity': I,
        'Hermitian': hermitian
    }

    def __init__(self, wires, gpu=False, **kwargs):
        super().__init__(wires=wires)

        if gpu:
            if not GPU_SUPPORTED:
                raise DeviceError(
                    'GPU not supported with installed version of qulacs. '
                    'Please install "qulacs-gpu" to use GPU simulation.'
                )

            self._state = QuantumStateGpu(wires)
        else:
            self._state = QuantumState(wires)

        self._circuit = QuantumCircuit(wires)
        self._first_operation = True

    def apply(self, operation, wires, par):
        if operation == 'BasisState' and not self._first_operation:
            raise DeviceError(
                'Operation {} cannot be used after other Operations have already been applied '
                'on a {} device.'.format(operation, self.short_name)
            )

        self._first_operation = False

        if operation == 'QubitStateVector':
            if len(par[0]) != 2**len(wires):
                raise ValueError('State vector must be of length 2**wires.')

            self._state.load(par[0])

            return
        elif operation == 'BasisState':
            if len(par[0]) != len(wires):
                raise ValueError('Basis state must prepare all qubits.')

            basis_state = 0
            for bit in reversed(par[0]):
                basis_state = (basis_state << 1) | bit

            self._state.set_computational_basis(basis_state)

            return
        elif operation == 'QubitUnitary':
            if len(par[0]) != 2 ** len(wires):
                raise ValueError('Unitary matrix must be of shape (2**wires, 2**wires).')

            unitary_gate = gate.DenseMatrix(wires, par[0])
            self._circuit.add_gate(unitary_gate)

            return
        elif operation in ('Rot', 'CRZ', 'Toffoli', 'CSWAP'):
            mapped_operation = self._operations_map[operation]
            if callable(mapped_operation):
                gate_matrix = mapped_operation(*par)
            else:
                gate_matrix = mapped_operation

            dense_gate = gate.DenseMatrix(wires, gate_matrix)
            self._circuit.add_gate(dense_gate)

            return

        mapped_operation = self._operations_map[operation]
        self._circuit.add_gate(mapped_operation(*wires, *par))

    @property
    def state(self):
        return self._state.get_vector()

    def pre_measure(self):
        self._circuit.update_quantum_state(self._state)

    def expval(self, observable, wires, par):
        bra = self._state.copy()

        if isinstance(observable, list):
            A = self._get_tensor_operator_matrix(observable, par)
            wires = [item for sublist in wires for item in sublist]
        else:
            A = self._get_operator_matrix(observable, par)

        dense_gate = gate.DenseMatrix(wires, A)
        dense_gate.update_quantum_state(self._state)

        expectation = inner_product(bra, self._state)

        return expectation.real

    def probabilities(self):
        states = itertools.product(range(2), repeat=self.num_wires)
        probs = np.abs(self.state)**2

        return OrderedDict(zip(states, probs))

    def reset(self):
        self._state.set_zero_state()
        self._circuit = QuantumCircuit(self.num_wires)

    def _get_operator_matrix(self, operation, par):
        A = self._observable_map[operation]
        if not callable(A):
            return A

        return A(*par)

    def _get_tensor_operator_matrix(self, obs, par):
        ops = [self._get_operator_matrix(o, p) for o, p in zip(obs, par)]
        return functools.reduce(np.kron, ops)
