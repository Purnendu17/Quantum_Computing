import numpy as np

# === Define common single-qubit gates ===
I = np.eye(2, dtype=complex)
X = np.array([[0, 1],
              [1, 0]], dtype=complex)
H = (1/np.sqrt(2)) * np.array([[1, 1],
                                [1, -1]], dtype=complex)
Z = np.array([[1, 0],
              [0, -1]], dtype=complex)

def apply_gate(state, gate, qubit, n_qubits):
    """
    Apply a single-qubit gate to 'qubit' in an n-qubit state.
    qubit = 0 is MSB, qubit = n-1 is LSB.
    """
    op = 1
    for i in range(n_qubits):
        if i == qubit:
            op = np.kron(op, gate)
        else:
            op = np.kron(op, I)
    return op @ state

# === Example usage: 3 qubits ===
n_qubits = 3

# Initial state |000>
state = np.zeros((2**n_qubits, 1), dtype=complex)
state[0,0] = 1

print("Initial state |000>:\n", state.flatten())

# Apply Hadamard to qubit 0 (MSB)
state = apply_gate(state, H, qubit=0, n_qubits=n_qubits)

# Apply X to qubit 2 (LSB)
state = apply_gate(state, X, qubit=2, n_qubits=n_qubits)

print("\nFinal state vector:\n", state.flatten())
