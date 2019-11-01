import numpy as np
import functools

from pennylane import Device, DeviceError

from qulacs import Observable, QuantumCircuit, QuantumState
from qulacs.gate import CNOT, RX, RY, RZ, X, Y, Z, H, U3, DenseMatrix
from qulacs.state import inner_product

from . import __version__


GPU_SUPPORTED = True
try:
    from qulacs import QuantumStateGpu
except ImportError:
    GPU_SUPPORTED = False


X_matrix = np.array([[0, 1], [1, 0]])
Y_matrix = np.array([[0, -1j], [1j, 0]])
Z_matrix = np.array([[1, 0], [0, -1]])
H_matrix = np.array([[1, 1], [1, -1]])/np.sqrt(2)


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
        'CNOT': CNOT,
        'RX': RX,
        'RY': RY,
        'RZ': RZ,
        'Rot': U3,
        'PauliX': X,
        'PauliY': Y,
        'PauliZ': Z,
        'Hadamard': H
    }
    _observable_map = {
        'PauliX': X_matrix,
        'PauliY': Y_matrix,
        'PauliZ': Z_matrix,
        'Hadamard': H_matrix,
        'Hermitian': hermitian
    }

    def __init__(self, wires, gpu=False, **kwargs):
        super().__init__(wires=wires)

        if gpu:
            if not GPU_SUPPORTED:
                raise DeviceError('GPU not supported with installed version of qulacs.' +
                                  ' Please install "qulacs-gpu" to use GPU simulation.')

            self._state = QuantumStateGpu(wires)
        else:
            self._state = QuantumState(wires)

        self._circuit = QuantumCircuit(wires)

    def apply(self, operation, wires, par):
        if operation == 'QubitStateVector':
            if len(par[0]) != 2**len(wires):
                raise ValueError('State vector must be of length 2**wires.')
            self._state.load(par[0])
            return

        mapped_operation = self._operations_map[operation]
        self._circuit.add_gate(mapped_operation(*wires, *par))

    @property
    def state(self):
        state = self._state.get_vector()

        return state.reshape([2] * self.num_wires).T.flatten()

    def pre_measure(self):
        self._circuit.update_quantum_state(self._state)

    def expval(self, observable, wires, par):
        bra = self._state.copy()

        if isinstance(observable, list):
            A = self._get_tensor_operator_matrix(observable, par)
        else:
            A = self._get_operator_matrix(observable, par)

        flat_wires = [item for sublist in wires for item in sublist]
        dense_gate = DenseMatrix(flat_wires, A)
        dense_gate.update_quantum_state(self._state)

        expectation = inner_product(bra, self._state)

        return expectation.real

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
